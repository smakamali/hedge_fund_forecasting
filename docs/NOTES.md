# Design and Development Notes 

# Questions:
1- Shouldn't the prediction made on the validation set using Model A' (pred_ap_val) be made sequentially? Or at least, shouldn't the sequential score be obtained additionally?
2- Shouldn't model B be trained only with the 90/10 train/val split, and not on the whole training? When it is trained on the full training including val, don't we have a signal leakage from predicitons of model B for 10% val data of model A'?