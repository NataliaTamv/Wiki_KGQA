| id | ημ/νία      | tag                     | endpoint | αρχείο                 | LLM                   | k | N   | F1   | P    | R    | ok/μερ/λάθ/κεν | λ  |
|----|-------------|-------------------------|----------|------------------------|-----------------------|---|-----|------|------|------|----------------|----|
| 1  | 19/11 18:27 | sample                  | endp1    | sample_questions.json  | llama3.1:8b-q8_0      | 3 | 2   | 0.00 | 0.00 | 0.00 | 0/0/1/1        | 1  |
| 2  | 19/11 20:20 | run_dev_all_questions   | endp1    | dev_questions_all.json | llama3.1:8b-q8_0      | 3 | 100 | 0.39 | 0.38 | 0.41 | 36/6/17/41     | 83 |
| 3  | 19/11 22:54 | failed_lqwen            | endp1    | dev_failed.json        | qwen2.5-coder:7b-q8_0 | 3 | 65  | 0.16 | 0.15 | 0.22 | 8/6/18/33      | 58 |
| 4  | 20/11 00:18 | dev_questions_all_lqwen | endp1    | dev_questions_all.json | qwen2.5-coder:7b-q8_0 | 3 | 100 | 0.40 | 0.40 | 0.44 | 38/6/22/34     | 75 |
| 5  | 21/11 19:14 | colab_qwen14b           | endp2    | dev_failed.json        | qwen2.5-coder:14b     | 3 | 20  | 0.33 | 0.33 | 0.35 | 6/1/7/6        | 5  |
| 6  | 21/11 19:30 | colab_qwen14b           | endp2    | dev_failed.json        | qwen2.5-coder:14b     | 3 | 65  | 0.25 | 0.24 | 0.27 | 13/6/17/29     | 15 |
| 7  | 21/11 20:01 | colab_qwen14b_all       | endp2    | dev_questions_all.json | qwen2.5-coder:14b     | 3 | 100 | 0.47 | 0.47 | 0.49 | 44/6/19/31     | 19 |
| 8  | 22/11 16:03 | colab_qwen14b_20        | endp2    | dev_questions_20.json  | qwen2.5-coder:14b     | 3 | 20  | 0.68 | 0.68 | 0.70 | 13/1/2/4       | 4  |
