import argparse

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def quantize(model_id, group_size, version):
    quant_path = model_id.replace("/", "-") + f"_{group_size}_{version}"
    quant_config = {"zero_point": True, "q_group_size": group_size, "w_bit": 4, "version": version}

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B", help="Model id or path")
    parser.add_argument("--group_size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--version", choices=["GEMM", "GEMV"], default="GEMM")
    args = parser.parse_args()
    quantize(args.model_id, args.group_size, args.version)
