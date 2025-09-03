from shared.config import load_config

from mlx_lm.utils import load
from mlx_lm.generate import generate

def main():
    config = load_config()

    # model, tokenizer = load(config['model']['repo_id'])
    model, tokenizer = load(config['model']['repo_id'], adapter_path=config['model']['output_location'], lazy=False)

    # prompt_user = input()
    prompt_user = "How do I cook spaghetti for dinner?"
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt_user},
    ]
    prompt = tokenizer.apply_chat_template( # type: ignore
        messages,
        tokenize=False
    ) 

    print(prompt)

    print("\n" + "###" * 10 + "\n")

    response = generate(model, tokenizer, prompt)
    print(response)

if __name__ == '__main__':
    main()
