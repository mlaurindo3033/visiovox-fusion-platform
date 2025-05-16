import requests, hashlib, os, argparse

def calculate_and_print_hash(model_name, model_url):
    print(f"{model_name}:", end=" ")
    sha256 = hashlib.sha256()
    temp_file_name = f"{model_name}_temp.onnx"
    try:
        r = requests.get(model_url, stream=True)
        r.raise_for_status()
        with open(temp_file_name, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    sha256.update(chunk)
        print(sha256.hexdigest())
        return sha256.hexdigest()
    except Exception as e:
        print(f"Error processing {model_name} from {model_url}: {e}")
        return None
    finally:
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model and compute its SHA256 hash.")
    parser.add_argument("--url", type=str, help="URL of the model to download and hash.")
    parser.add_argument("--name", type=str, help="Name of the model (used for printing and temporary file name if URL is provided). Defaults to 'model_from_url'.")
    args = parser.parse_args()

    if args.url:
        model_name = args.name if args.name else "model_from_url"
        calculate_and_print_hash(model_name, args.url)
    else:
        print("No URL provided, running for pre-defined models:")
        models = {
            "gfpgan_1.4": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
            "codeformer": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
            "simswap_256": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/simswap_256.onnx"
        }
        for name, url in models.items():
            calculate_and_print_hash(name, url) 