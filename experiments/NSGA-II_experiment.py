import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing as mp
import os
import time
import pickle
import json
import sys
import subprocess
import threading
import queue
import psutil
from contextlib import contextmanager
import signal
import gc

from experiment_utils import Experiment, prepare_experiment_dirs 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('FATAL')

os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

def setup_deap_classes():
    """Setup DEAP classes - call this once"""
    if not hasattr(creator, "FitnessMulti"):
        weights = (1.0, 1.0, 1.0, 1.0)  # 1 security + 3 cost metrics
        creator.create("FitnessMulti", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

# Call setup once at module level
setup_deap_classes()

model_dir, results_dir, run_tag = prepare_experiment_dirs("NSGAII")
create_experiment_snapshots([25, 50, 75, 100])

# Environment and agent settings
static_features = [
    "host_compromise_ratio",
    "attack_path_exposure",
    "overall_asr_avg",
    "roa",
    "risk",
]

time_features = [
    "mtd_freq",
    "overall_mttc_avg",
    "time_since_last_mtd",
    "shortest_path_variability",
    "ip_variability",
    "attack_type",
    "downtime_ratio",
    "agent_time_ratio",
    "mtd_action_ratio",
]

features = {"static": static_features, "time": time_features}
state_size = 5
time_series_size = 9
action_size = 5

# --- Search space ---
HYPERPARAM_SPACE = {
    "gamma": (0.0, 1.0),          
    "epsilon": (0.0, 1.0),        
    "epsilon_decay": (0.0, 1.0),  
    "train_start": (500, 5000),   
}

SECURITY_METRICS = ["time_to_compromise"]
COST_METRICS = ["mtd_action_ratio", "agent_time_ratio", "downtime_ratio"]

# --- System Resource Management ---
def get_optimal_worker_count():
    return 8

# --- Simple Cleanup Functions ---
def perform_batch_cleanup():
    """
    Simple cleanup between batches focused on CPU resources.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Starting batch cleanup...")
    cleanup_start = time.time()
    
    # 1. Force garbage collection
    print("  • Running garbage collection...")
    collected = gc.collect()
    if collected > 0:
        print(f"    - Collected {collected} objects")
    
    # 2. Clear TensorFlow session
    print("  • Clearing TensorFlow sessions...")
    try:
        tf.keras.backend.clear_session()
    except Exception as e:
        print(f"    - TF cleanup warning: {e}")
    
    # 3. Kill any orphaned training processes
    print("  • Checking for orphaned processes...")
    try:
        current_pid = os.getpid()
        killed_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
            try:
                if (proc.info['name'] and 'python' in proc.info['name'].lower() and 
                    proc.info['cmdline'] and any('training_script.py' in str(cmd) for cmd in proc.info['cmdline']) and
                    proc.info['ppid'] != current_pid and proc.info['pid'] != current_pid):
                    proc.kill()
                    killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if killed_count > 0:
            print(f"    - Killed {killed_count} orphaned processes")
    except Exception as e:
        print(f"    - Process cleanup warning: {e}")
    
    cleanup_time = time.time() - cleanup_start
    print(f"  • Cleanup completed in {cleanup_time:.2f} seconds")

def wait_between_batches(seconds):
    """
    Simple wait with countdown.
    """
    print(f"Waiting {seconds} seconds before next batch...")
    for i in range(seconds):
        remaining = seconds - i
        print(f"\r  ⏳ {remaining}s remaining...", end="", flush=True)
        time.sleep(1)
    print(f"\r  ✅ Wait complete!           ")

# --- Helpers ---
def individual_to_params(individual):
    params = {}
    for i, key in enumerate(HYPERPARAM_SPACE.keys()):
        low, high = HYPERPARAM_SPACE[key]
        if key == "train_start":
            val = int(round(individual[i] / 500) * 500)
            params[key] = max(low, min(high, val))
        else:
            val = max(low, min(high, individual[i]))
            params[key] = float(val)
    return params

# --- Robust subprocess training with better error handling ---
def train_single_candidate_subprocess_robust(params_dict):
    """
    Robust subprocess training with comprehensive error handling and monitoring.
    """
    candidate_id = params_dict.get('candidate_id', -1)
    process_start_time = time.time()
    process = None
    
    try:
        params_dict['model_dir'] = model_dir
        params_json = json.dumps(params_dict, default=str)
        cmd = [sys.executable, 'training_script.py', params_json]
        
        print(f"[{time.strftime('%H:%M:%S')}] Starting candidate {candidate_id}")
        
        # Enhanced environment with strict resource limits
        env = os.environ.copy()
        env.update({
            'PYTHONIOENCODING': 'utf-8:replace',
            'PYTHONUNBUFFERED': '1',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'TF_NUM_INTEROP_THREADS': '1',
            'TF_NUM_INTRAOP_THREADS': '1',
            'TF_CPP_MIN_LOG_LEVEL': '3',
            'CUDA_VISIBLE_DEVICES': '',
            'TF_FORCE_GPU_ALLOW_GROWTH': 'false',
        })
        
        # Create process with better isolation
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            creation_flags = 0
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=os.getcwd(),
            env=env,
            shell=False,
            startupinfo=startupinfo,
            creationflags=creation_flags
        )
        
        # Monitor process with timeout and resource checking
        timeout_seconds = 5400  # 90 minutes
        check_interval = 30  # Check every 30 seconds
        
        while True:
            try:
                # Check if process has completed
                return_code = process.poll()
                if return_code is not None:
                    # Process completed
                    stdout, stderr = process.communicate(timeout=10)
                    break
                
                # Check timeout
                elapsed = time.time() - process_start_time
                if elapsed > timeout_seconds:
                    print(f"[{time.strftime('%H:%M:%S')}] Candidate {candidate_id} timeout after {elapsed:.1f}s")
                    _kill_process_tree(process)
                    return candidate_id, None, params_dict, "Training timeout"
                
                # Check memory usage
                try:
                    proc = psutil.Process(process.pid)
                    memory_mb = proc.memory_info().rss / 1024 / 1024
                    if memory_mb > 6144:  # 6GB limit
                        print(f"[{time.strftime('%H:%M:%S')}] Candidate {candidate_id} exceeded memory limit: {memory_mb:.1f}MB")
                        _kill_process_tree(process)
                        return candidate_id, None, params_dict, "Memory limit exceeded"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                # Wait before next check
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print(f"[{time.strftime('%H:%M:%S')}] Candidate {candidate_id} interrupted by user")
                _kill_process_tree(process)
                return candidate_id, None, params_dict, "User interrupt"
        
        elapsed_time = time.time() - process_start_time
        print(f"[{time.strftime('%H:%M:%S')}] Candidate {candidate_id} completed in {elapsed_time:.1f}s")
        
        if return_code != 0:
            error_msg = f"Failed with code {return_code}"
            if stderr:
                error_msg += f": {stderr[:200]}..."
            return candidate_id, None, params_dict, error_msg
        
        # Parse successful result
        try:
            output_lines = stdout.strip().split('\n')
            result_line = None
            
            for line in output_lines:
                if line.startswith('RESULT:'):
                    result_line = line[7:].strip()
                    break
            
            if result_line is None:
                return candidate_id, None, params_dict, "No RESULT found in output"
            
            result_data = json.loads(result_line)
            
            if result_data['success']:
                model_name = result_data['model_name']
                params = result_data['params']
                print(f"Successfully trained candidate {candidate_id}: {model_name}")
                return candidate_id, model_name, params, None
            else:
                error_msg = result_data.get('error', 'Unknown training error')
                return candidate_id, None, params_dict, error_msg
                
        except json.JSONDecodeError as e:
            return candidate_id, None, params_dict, f"JSON decode error: {str(e)}"
            
    except Exception as e:
        elapsed_time = time.time() - process_start_time
        print(f"Exception in training candidate {candidate_id} after {elapsed_time:.1f}s: {str(e)}")
        if process:
            _kill_process_tree(process)
        return candidate_id, None, params_dict, str(e)

def _kill_process_tree(process):
    """Kill process and all its children"""
    try:
        if os.name == 'nt':
            # Windows
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], 
                         capture_output=True, timeout=10)
        else:
            # Unix
            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.kill()
                parent.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        print(f"Error killing process tree: {e}")

# --- Simple Batch-based training ---
def train_population_batch(population, generation=0, max_workers=None, cleanup_wait=30):
    """
    Train population in batches with cleanup between batches.
    Simple implementation focused on CPU resource management.
    """
    if max_workers is None:
        max_workers = get_optimal_worker_count()
    
    print(f"\n=== Batch Training Generation {generation} ===")
    print(f"Training {len(population)} candidates in batches of {max_workers}")
    print(f"Cleanup wait time: {cleanup_wait} seconds")
    
    # Convert individuals to parameter dictionaries
    params_list = []
    for i, individual in enumerate(population):
        params = individual_to_params(individual)
        params['candidate_id'] = i
        params['generation'] = generation
        params_list.append(params)
    
    # Split into batches
    batches = []
    for i in range(0, len(params_list), max_workers):
        batch = params_list[i:i + max_workers]
        batches.append(batch)
    
    print(f"Split into {len(batches)} batches")
    
    training_results = {}
    failed_candidates = []
    total_start_time = time.time()
    
    # Process each batch
    for batch_idx, batch_params in enumerate(batches):
        batch_start_time = time.time()
        print(f"\n--- Processing Batch {batch_idx + 1}/{len(batches)} ---")
        print(f"Batch size: {len(batch_params)} candidates")
        print(f"Candidates: {[p['candidate_id'] for p in batch_params]}")
        
        # Train current batch using ThreadPoolExecutor
        batch_results = {}
        batch_failures = []
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all training tasks in current batch
                future_to_id = {
                    executor.submit(train_single_candidate_subprocess_robust, params): params['candidate_id']
                    for params in batch_params
                }
                
                print(f"Submitted {len(future_to_id)} training tasks for batch {batch_idx + 1}")
                
                # Collect results for current batch
                completed_count = 0
                for future in as_completed(future_to_id):
                    candidate_id = future_to_id[future]
                    try:
                        result_id, model_name, params, error_msg = future.result(timeout=6000)  # 100 min timeout
                        
                        if model_name is not None:
                            batch_results[result_id] = (model_name, params)
                            completed_count += 1
                            print(f"✓ Batch {batch_idx + 1}: Completed {completed_count}/{len(batch_params)} (Candidate {result_id})")
                        else:
                            batch_failures.append((result_id, error_msg or "Unknown error"))
                            batch_results[result_id] = (None, next(p for p in batch_params if p['candidate_id'] == result_id))
                            print(f"✗ Batch {batch_idx + 1}: Failed candidate {result_id}: {error_msg}")
                            
                    except Exception as exc:
                        print(f"Exception collecting result for candidate {candidate_id} in batch {batch_idx + 1}: {exc}")
                        batch_failures.append((candidate_id, str(exc)))
                        batch_results[candidate_id] = (None, next(p for p in batch_params if p['candidate_id'] == candidate_id))
                        
        except Exception as e:
            print(f"Critical error in batch {batch_idx + 1} thread pool: {e}")
            # Ensure all candidates in batch have results
            for params in batch_params:
                cid = params['candidate_id']
                if cid not in batch_results:
                    batch_results[cid] = (None, params)
                    batch_failures.append((cid, "Thread pool error"))
        
        # Add batch results to overall results
        training_results.update(batch_results)
        failed_candidates.extend(batch_failures)
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        successful_in_batch = len([r for r in batch_results.values() if r[0] is not None])
        
        print(f"✅ Batch {batch_idx + 1} completed in {batch_duration:.1f} seconds")
        print(f"   Success: {successful_in_batch}/{len(batch_params)} candidates")
        print(f"   Failures: {len(batch_failures)} candidates")
        
        # Perform cleanup and wait between batches (except after the last batch)
        if batch_idx < len(batches) - 1:  # Don't cleanup after the last batch
            print(f"\n🧹 Starting cleanup after batch {batch_idx + 1}...")
            
            # Brief wait for processes to fully terminate
            time.sleep(5)
            
            # Perform batch cleanup
            perform_batch_cleanup()
            
            # Wait before next batch
            wait_between_batches(cleanup_wait)
            
        else:
            print(f"\n🏁 All batches completed - skipping final cleanup")
    
    # Final summary
    end_time = time.time()
    total_duration = end_time - total_start_time
    success_count = len([r for r in training_results.values() if r[0] is not None])
    
    print(f"\n{'='*60}")
    print(f"Batch training completed!")
    print(f"Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Successfully trained: {success_count}/{len(population)} candidates")
    print(f"Failed candidates: {len(failed_candidates)}")
    print(f"Batches processed: {len(batches)}")
    print(f"Average time per batch: {total_duration/len(batches):.1f} seconds")
    print(f"{'='*60}")
    
    return training_results

# --- Evaluation functions remain the same ---
def evaluate_single_candidate_sequential(model_name, params, candidate_id):
    """Sequential evaluation function"""
    try:
        print(f"Evaluating candidate {candidate_id}: {model_name}")
        
        exp = Experiment(
            epsilon=params['epsilon'],
            start_time=0,
            finish_time=15000,
            mtd_interval=[200],
            network_size=[150],
            total_nodes=150,
            new_network=True,
            model=model_name,
            trial=100,
            result_head_path="",
            file_name=model_name,
            model_dir=model_dir,
            results_dir=results_dir,
            run_tag=run_tag
        )
        exp.run_trials('mtd_ai')

        df = exp.get_result_checkpoint_median('mtd_ai', model_name, checkpoints=9)
        averages = df.mean()

        fitness = []
        for m in SECURITY_METRICS:
            fitness.append(averages[m])
        for m in COST_METRICS:
            fitness.append(-averages[m])

        print(f"Evaluation complete for candidate {candidate_id}: {model_name}")
        return candidate_id, tuple(fitness)
        
    except Exception as e:
        print(f"Error evaluating candidate {candidate_id} ({model_name}): {str(e)}")
        worst_fitness = tuple([float('-inf')] * (len(SECURITY_METRICS) + len(COST_METRICS)))
        return candidate_id, worst_fitness

def evaluate_population_sequential(population, training_results):
    """Evaluate population sequentially"""
    print(f"\n=== Sequential Evaluation ===")
    print(f"Evaluating {len(population)} candidates")
    
    successful_evaluations = 0
    
    for i, individual in enumerate(population):
        model_name, params = training_results[i]
        
        if model_name is None:
            print(f"Skipping evaluation for candidate {i} due to training failure")
            worst_fitness = tuple([float('-inf')] * (len(SECURITY_METRICS) + len(COST_METRICS)))
            individual.fitness.values = worst_fitness
            continue
        
        try:
            candidate_id, fitness = evaluate_single_candidate_sequential(model_name, params, i)
            individual.fitness.values = fitness
            successful_evaluations += 1
            print(f"✓ Completed evaluation {successful_evaluations}")
        except Exception as e:
            print(f"✗ Evaluation failed for candidate {i}: {e}")
            worst_fitness = tuple([float('-inf')] * (len(SECURITY_METRICS) + len(COST_METRICS)))
            individual.fitness.values = worst_fitness
    
    print(f"Sequential evaluation completed: {successful_evaluations} successful")
    return population

# --- Parameter generation ---
def generate_discretized_uniform(low, high, precision=3):
    return round(random.uniform(low, high), precision)

def mutPolynomialBoundedRounded(individual, eta, low, up, indpb, precision=3):
    size = len(individual)
    if not isinstance(low, list):
        low = [low] * size
    if not isinstance(up, list):
        up = [up] * size
    
    for i in range(size):
        if random.random() <= indpb:
            x = individual[i]
            xl = low[i]
            xu = up[i]
            
            if xu - xl < 1e-14:
                continue
                
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.0)
            
            if rand <= 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                delta_q = 1.0 - (val ** mut_pow)
            
            x = x + delta_q * (xu - xl)
            
            if isinstance(x, complex):
                x = x.real
            x = max(xl, min(xu, x))
            
            if i < 3:  # continuous parameters
                individual[i] = round(float(x), precision)
            else:  # train_start
                individual[i] = int(round(x / 500) * 500)
    
    return individual,

# --- DEAP setup ---
toolbox = base.Toolbox()
toolbox.register("attr_gamma", generate_discretized_uniform, *HYPERPARAM_SPACE["gamma"], 3)
toolbox.register("attr_epsilon", generate_discretized_uniform, *HYPERPARAM_SPACE["epsilon"], 3)
toolbox.register("attr_epsilon_decay", generate_discretized_uniform, *HYPERPARAM_SPACE["epsilon_decay"], 3)
toolbox.register("attr_train_start", random.randrange, 500, 5001, 500)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_gamma, toolbox.attr_epsilon,
                  toolbox.attr_epsilon_decay, toolbox.attr_train_start), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutPolynomialBoundedRounded,
                 low=[HYPERPARAM_SPACE["gamma"][0], HYPERPARAM_SPACE["epsilon"][0],
                      HYPERPARAM_SPACE["epsilon_decay"][0], HYPERPARAM_SPACE["train_start"][0]],
                 up=[HYPERPARAM_SPACE["gamma"][1], HYPERPARAM_SPACE["epsilon"][1],
                     HYPERPARAM_SPACE["epsilon_decay"][1], HYPERPARAM_SPACE["train_start"][1]],
                 eta=0.5, indpb=0.2, precision=3)
toolbox.register("select", tools.selNSGA2)

# --- Main NSGA-II algorithm with simple batch processing ---
def run_nsga2_batch(toolbox, pop_size, generations, max_workers=None, 
                   cxpb=0.6, mutpb=0.3, verbose=True, cleanup_wait=30):
    """
    NSGA-II with simple batch processing and cleanup between batches.
    """
    # Initialize population
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "avg", "std", "min", "max"
    
    if max_workers is None:
        max_workers = get_optimal_worker_count()
    
    print(f"\n{'='*60}")
    print(f"NSGA-II Hyperparameter Optimization (Batch Processing)")
    print(f"Population: {pop_size}, Generations: {generations}")
    print(f"Batch size: {max_workers} candidates per batch")
    print(f"Cleanup wait: {cleanup_wait} seconds between batches")
    print(f"{'='*60}")
    
    # Generation 0 - Initial Population
    print(f"\nGeneration 0 (Initial Population)")
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    
    # Use batch training for initial population
    training_results = train_population_batch(invalid_ind, 0, max_workers, cleanup_wait)
    evaluate_population_sequential(invalid_ind, training_results)
    hof.update(pop)
    
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    if verbose:
        print(f"Stats: {logbook.stream}")
    
    # Subsequent Generations
    for gen in range(1, generations + 1):
        print(f"\nGeneration {gen}")
        
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        if invalid_ind:
            # Use batch training for offspring
            training_results = train_population_batch(invalid_ind, gen, max_workers, cleanup_wait)
            evaluate_population_sequential(invalid_ind, training_results)
        
        # Selection
        pop = toolbox.select(pop + offspring, pop_size)
        hof.update(pop)
        
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            print(f"Stats: {logbook.stream}")
    
    return pop, logbook, hof

if __name__ == "__main__":
    # Configuration
    pop_size = 16
    generations = 5
    cleanup_wait = 30  # seconds between batches
    
    print("NSGA-II Hyperparameter Optimization (Simple Batch Processing)")
    print(f"Population Size: {pop_size}")
    print(f"Generations: {generations}")
    print(f"Cleanup Wait: {cleanup_wait}s")
    
    start_time = time.time()
    
    try:
        final_pop, logbook, hof = run_nsga2_batch(
            toolbox, 
            pop_size=pop_size, 
            generations=generations,
            max_workers=None,  # Will use get_optimal_worker_count()
            cxpb=0.6, 
            mutpb=0.3,
            verbose=True,
            cleanup_wait=cleanup_wait
        )
        
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Pareto front size: {len(hof)}")
    print(f"{'='*60}")

    # Analyze results
    pareto_data = []
    for ind in hof:
        params = individual_to_params(ind)
        fit_vals = ind.fitness.values
        sec_vals = fit_vals[0:len(SECURITY_METRICS)]
        cost_vals = [-v for v in fit_vals[len(SECURITY_METRICS):]]
        pareto_data.append({**params,
                            **{m: s for m, s in zip(SECURITY_METRICS, sec_vals)},
                            **{m: c for m, c in zip(COST_METRICS, cost_vals)}})

    pareto_df = pd.DataFrame(pareto_data)

    if not pareto_df.empty:
        # Best security
        best_security_val = pareto_df[SECURITY_METRICS[0]].max()
        best_security = pareto_df[pareto_df[SECURITY_METRICS[0]] == best_security_val]

        # Best cost
        pareto_df["total_cost"] = pareto_df[COST_METRICS].sum(axis=1)
        best_cost_val = pareto_df["total_cost"].min()
        best_cost = pareto_df[pareto_df["total_cost"] == best_cost_val]

        # Balanced
        tol = 0.02
        balanced = pareto_df[
            pareto_df[SECURITY_METRICS[0]] >= best_security_val * (1 - tol)
        ].sort_values("total_cost", ascending=True).head()

        print("\n=== RESULTS ===")
        print("\nBest Security Solution:")
        print(best_security.to_string(index=False))
        print("\nBest Cost Solution:")
        print(best_cost.to_string(index=False))
        print("\nBalanced Solutions:")
        print(balanced.to_string(index=False))
        
        # Save results
        results_file = f"nsga2_batch_results_{int(time.time())}.csv"
        pareto_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
        
    else:
        print("\nNo valid Pareto solutions found!")
        
    print(f"\nNSGA-II batch optimization completed successfully!")