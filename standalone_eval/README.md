VLEP Evalation
================================================================

### Task Definition
Given a video with associated dialogue as premise, and two possible future events, 
the VLEP task requires systems to predict which one is more likely to happen. 
The task performance is evaluated using accuracy.


### How to construct a prediction file?

A prediction file is `.jsonl` file. Each line in this file contains a single json string that 
can be loaded as a `dict` with two entries. 
```
{"example_id": int, "pred_ans": int}
``` 
`example_id` is the id of the example, `pred_ans` is the index of the predicted answer, in `{0, 1}`. 
 
### Run Evaluation
At project root, run
```
bash standalone_eval/eval_sample.sh 
```
This command will use [eval.py](eval.py) to evaluate the provided `sample_dev_submission.json` file, 
the output will be written into `sample_dev_submission_metrics_new.json`. 
Its content should be similar if not the same as `sample_dev_submission_metrics.json` file.


### Codalab Submission
To get your model's performance on `test` split, 
please submit both `dev` and `test` predictions to our 
[CodaLab evaluation server](https://competitions.codalab.org/competitions/26881). 
The submission file should be a single `.zip ` file (no enclosing folder) 
that contains the two prediction files 
`vlep_test_submission.json` and `vlep_dev_submission.json`, each of the `*submission.json` file 
should be formatted as instructed above. 
