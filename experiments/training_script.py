#!/usr/bin/env python3
import sys, os, io, contextlib, json, random, numpy as np, traceback

# Setup to ensure performance and avoid crashes due to unwanted output during training
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MAX_VMAP_VECTORIZATION_BATCH_SIZE'] = '32'

import tensorflow as tf
tf.get_logger().setLevel('FATAL')
tf.config.optimizer.set_jit(True)  
THREADS_PER_PROC = int(os.environ.get("TF_THREADS_PER_PROC", "4"))

tf.config.threading.set_intra_op_parallelism_threads(THREADS_PER_PROC)
tf.config.threading.set_inter_op_parallelism_threads(THREADS_PER_PROC)
tf.keras.backend.set_floatx('float32') 
tf.keras.mixed_precision.set_global_policy('float32')

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from run import execute_ai_training

def generate_model_name(params):
    """Generate universal model name from parameters with optional algorithm name"""
    
    # Extract algorithm name if present
    algorithm_name = params.pop('algorithm_name', None)
    
    # Remove non-parameter fields
    excluded_keys = {'candidate_id', 'generation', 'model_dir'}
    param_dict = {k: v for k, v in params.items() if k not in excluded_keys}
    
    # Create parameter string with parameter names - format values to avoid long unreadable decimals 
    param_parts = []
    for key, value in sorted(param_dict.items()): 
        if isinstance(value, float):
            formatted_value = f"{value:.3f}".rstrip('0').rstrip('.')
        else:
            formatted_value = str(value)
        param_parts.append(f"{key.upper()}_{formatted_value}")
    
    # Combine parts
    if algorithm_name:
        model_name = f"{algorithm_name}_{'_'.join(param_parts)}"
    else:
        model_name = '_'.join(param_parts)
    
    return model_name

def train_candidate(params_json_str):
    candidate_id = -1
    try:
        # Parse parameters
        params = json.loads(params_json_str)
        candidate_id = params.pop('candidate_id', 0)
        generation = params.pop('generation', 0)
        model_dir = params.pop('model_dir', './models')

        # Extract training parameters
        gamma = params.get('gamma', 0.95)
        epsilon = params.get('epsilon', 1.0)
        epsilon_decay = params.get('epsilon_decay', 0.995)
        train_start = params.get('train_start', 2000)
        batch_size = params.get('batch_size', 32)
        epsilon_min = params.get('epsilon_min', 0.01)

        static_features = [
            "host_compromise_ratio", "attack_path_exposure", "overall_asr_avg", "roa", "risk",
        ]
        time_features = [
            "mtd_freq", "overall_mttc_avg", "time_since_last_mtd", "shortest_path_variability",
            "ip_variability", "attack_type", "downtime_ratio", "agent_time_ratio", "mtd_action_ratio",
        ]
        features = {"static": static_features, "time": time_features}
        state_size = 5
        time_series_size = 9
        action_size = 5

        # Generate universal model name
        model_name = generate_model_name(params.copy())
        print(f"Training {model_name} (PID: {os.getpid()})")

        # Unique seed
        seed = hash((os.getpid(), candidate_id, generation)) % (2**31)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        if os.name == 'nt': # Windows
            import msvcrt
            stdout_fd = sys.stdout.fileno()
            stderr_fd = sys.stderr.fileno()
            
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            
            stdout_dup = os.dup(stdout_fd)
            stderr_dup = os.dup(stderr_fd)
            
            try:
                os.dup2(devnull_fd, stdout_fd)
                os.dup2(devnull_fd, stderr_fd)
                
                execute_ai_training(
                    gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                    batch_size=batch_size, train_start=int(train_start), start_time=0, finish_time=15000,
                    mtd_interval=[200], features=features, state_size=state_size,
                    time_series_size=time_series_size, action_size=action_size,
                    scheme="mtd_ai", total_nodes=150, new_network=True, episodes=100,
                    file_name=model_name, model_dir=model_dir
                )
            finally:
                # Restore file descriptors
                os.dup2(stdout_dup, stdout_fd)
                os.dup2(stderr_dup, stderr_fd)
                os.close(stdout_dup)
                os.close(stderr_dup)
                os.close(devnull_fd)
        else: # Unix
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    execute_ai_training(
                        gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                        batch_size=batch_size, train_start=int(train_start), start_time=0, finish_time=15000,
                        mtd_interval=[200], features=features, state_size=state_size,
                        time_series_size=time_series_size, action_size=action_size,
                        scheme="mtd_ai", total_nodes=150, new_network=True, episodes=100,
                        file_name=model_name, model_dir=model_dir
                    )

        result = {
            'success': True, 'candidate_id': candidate_id, 'model_name': model_name,
            'params': {'gamma': gamma, 'epsilon': epsilon, 'epsilon_decay': epsilon_decay, 
                      'train_start': train_start, 'batch_size': batch_size, 'epsilon_min': epsilon_min}
        }
        print(f"Training completed successfully: {model_name}")
        return json.dumps(result)

    except Exception as e:
        error_result = {
            'success': False, 'candidate_id': candidate_id,
            'error': str(e), 'traceback': traceback.format_exc()
        }
        print(f"Training failed for candidate {candidate_id}: {str(e)}")
        return json.dumps(error_result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python training_script.py <params_json>")
        sys.exit(1)

    params_json = sys.argv[1]
    result_json = train_candidate(params_json)
    print("RESULT:", result_json)