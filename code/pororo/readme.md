### how to install pororo (colab)

1. git clone pororo repository
    ```
    !git clone https://github.com/kakaobrain/pororo
    ```
2. change code
   1. setup.py
      ```
       #setup.py
       torch==1.6.0을 torch>=1.6.0으로 수정
       torchvision==0.7.0을 torchvision>=0.7.0으로 수정
       ```
   2. pororo/tasks/utils/tokenizer.py
      ```
       # pororo/tasks/utils/tokenizer.py
       class CustomTokenizer(BaseTokenizer):
          def __init__(
              ...
              tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
                 replacement=replacement,
                 prepend_scheme="first",
                 split=True,
              )

              tokenizer.decoder = decoders.Metaspace(
                 replacement=replacement,
                 prepend_scheme="first",
                 split=True,
              )
       ```
3.cd pororo   

4.pip install "pip<24.1"
