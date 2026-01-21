## JSSODa

1. `llm-jp/JSSODa` よりメタデータを作成
    - `src/data_gen/elements_jssoda.py`
        - train:
            ```
            python src/data_gen/elements_jssoda.py
            ```
        - validation: 
            ```
            python src/data_gen/elements_jssoda.py --split validation --output_dir data/JSSODa_validation
            ```
2. 画像を挿入する場所を決める
    - `src/data_gen/img_match_jssoda.py`
        - train: 
            ```
            python src/data_gen/img_match_jssoda.py
            ```
        - validation: 
            ```
            python src/data_gen/img_match_jssoda.py --split validation --input_file_path data/JSSODa_validation/jssoda_validation_text_elements.jsonl --output_dir data/JSSODa_validation
            ```
3. 画像を生成するためのプロンプトを生成
    - `src/data_gen/gen_prompt_jssoda.py`
4. 画像を生成
    - `src/data_gen/gen_img_jssoda.py`
5. 画像のキャプションを生成
    - `src/data_gen/gen_caption_jssoda.py`
6. タイトルを生成
    - `src/data_gen/gen_title_jssoda.py`