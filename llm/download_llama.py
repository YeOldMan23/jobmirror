"""
Install LLama model and test with dummy round
also provide token/s to determine the speed of the model
"""
from llama_cpp import Llama
import os
import time

def install_model(cache_dir : str,
                  model_id : str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", 
                  filename : str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
    """
    Install the model from Llama, default Mistral7B
    """
    print(f"Checking for model '{filename}' from '{model_id}'...")
        
    llm = Llama.from_pretrained(
        repo_id=model_id,
        filename=filename,
        verbose=False,  # Set to True for more detailed download progress
        **({'cache_dir': cache_dir} if cache_dir else {})
    )
    
    print(f"Model '{filename}' successfully loaded.")
    return llm


def test_model(llm_model : Llama):
    """
    Test the model to see if the output is available, and check token per second
    """

    prompt = "Count from the number 1 to 100, return just the numbers with spaces inbetween"
    max_token = 250

    starttime = time.time_ns()

    output = llm_model(
        prompt,
        max_tokens=max_token
        echo=False,
        stop=["\n"]
    )

    endtime = time.time_ns()

    # Check the output length and sizes
    if "usage" in output and "completion_tokens" in output["usage"]:
        tokens_generated = output["usage"]["completion_tokens"]
        duration = float(endtime - starttime) / (10 ** 9)
        
        if duration > 0 and tokens_generated > 0:
            tps = tokens_generated / duration
            print(f"Output Size : {tokens_generated}")
            print(f"Time taken : {duration:6f} seconds")
            print(f"Token Per Second : {tps}")
    pass

if __name__ == "__main__":
    home_dir = os.getcwd()

    print("--Installing Model--")
    cache_dir = os.path.join(home_dir, "model_cache")

    print("--Starting Model Test--")
    # Prepare the model
    llm_model = install_model(cache_dir)

    # Test the model
    test_model(llm_model)

    # Final output
    print("Model Installation and Test Complete!")
    
    