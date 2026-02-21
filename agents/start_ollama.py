import subprocess
import time
import socket
import os
import shlex

def is_port_in_use(port: int) -> bool:
    """Check if a TCP port is in use on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except Exception:
            return False

def kill_ollama_servers(port):
    """Kill all processes running on the specified ports."""
    if is_port_in_use(port):
        try:
            subprocess.check_call(["fuser", "-k", f"{port}/tcp"])
            print(f"Successfully killed processes on port {port}.")
        except subprocess.CalledProcessError:
            print(f"Failed to kill process on port {port}. It may not be running.")

def start_ollama_server(
    port: int,
    rank: int,
    logdir: str,
    global_rank: int,
    retries: int = 3,
    delay: int = 5,
    ollama_bin: str = None,
    ollama_models_dir: str = None,
):
    """Start a fresh Ollama server instance with retry logic."""
    ollama_bin = ollama_bin or os.environ.get("OLLAMA_BIN", "ollama")
    ollama_models_dir = ollama_models_dir or os.environ.get("OLLAMA_MODELS")

    for attempt in range(1, retries + 1):
        env_parts = [
            f"CUDA_VISIBLE_DEVICES={rank}",
            f"OLLAMA_HOST=127.0.0.1:{port}",
            "OLLAMA_DEBUG=1",
            "OLLAMA_MAX_LOADED_MODELS=1",
            "OLLAMA_NUM_PARALLEL=1",
        ]
        if ollama_models_dir:
            env_parts.append(f"OLLAMA_MODELS={shlex.quote(ollama_models_dir)}")

        ollama_cmd = shlex.quote(ollama_bin)
        log_path = shlex.quote(f"{logdir}/ollama_server_rank{global_rank}.log")
        cmd = f"{' '.join(env_parts)} nohup {ollama_cmd} serve > {log_path} 2>&1 &"
        print(f"Attempt {attempt}: Starting Ollama server on port {port} & rank {global_rank} with command:\n{cmd}")
        proc = subprocess.Popen(cmd, shell=True)
        
        # Wait a bit for the server to initialize.
        time.sleep(delay)
        
        if is_port_in_use(port):
            print(f"Ollama server successfully started on port {port}.")
            return proc
        else:
            print(f"Attempt {attempt} failed to start Ollama server on port {port}.")
    
    print("Failed to start Ollama server after multiple retries. Exiting.")
    return None

def stop_ollama_server(proc, port):
    """Stop a specific Ollama server process."""
    if proc:
        print(f"Stopping Ollama server on port {port}...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Force killing Ollama server on port {port}.")
            proc.kill()
    kill_ollama_servers(port)  # Ensure cleanup
