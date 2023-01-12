import torch

# GPU 사용 가능 -> True, GPU 사용 불가 -> False

if __name__ == "__main__":
    print(torch.cuda.is_available())