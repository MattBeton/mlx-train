from mlx.utils import tree_flatten, tree_reduce
from shared.config import load_config

from mlx_lm.utils import load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler

def main():
    config = load_config()

    # model, tokenizer = load(config['model']['repo_id'])
    model, tokenizer = load(config['model']['repo_id'], adapter_path=config['model']['output_location'], lazy=False)

    # print([(x,y.sum()) for x,y in tree_flatten(model)])

    prompt_user = input(">>> ")
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt_user},
    ]
    prompt = tokenizer.apply_chat_template( # type: ignore
        messages,
        tokenize=False
    ) 

    print("\n" + "###" * 10 + "\n")

    sampler = make_sampler(temp=0.8)
    response = generate(model, tokenizer, prompt, sampler=sampler)
    print(response)

if __name__ == '__main__':
    main()
