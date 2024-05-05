rm -r ./models 
rm -r ./lightning_logs 
mv ./results.csv ./results_last_experiments.csv
# gamma=0.1
python main.py --config config/pair_train_gamma_0.1_fgsm_config.yaml fit

python main.py --config config/pair_train_gamma_0.1_fgsm_config.yaml test --model.adv_test_method=idAttack

python main.py --config config/pair_train_gamma_0.1_fgsm_config.yaml test \
                --model.adv_test_method=FGSMAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.eps=0.031

python main.py --config config/pair_train_gamma_0.1_fgsm_config.yaml test \
                --model.adv_test_method=PGDAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.alpha=0.031 \
                --model.adv_test_method.eps=0.008

python main.py --config config/pair_train_gamma_0.1_fgsm_config.yaml test \
                --model.adv_test_method=DeepfoolAttack \
                --model.adv_test_method.max_iters=20 \
                --model.adv_test_method.overshoot=0.02

# gamma=0.5
python main.py --config config/pair_train_gamma_0.5_fgsm_config.yaml fit

python main.py --config config/pair_train_gamma_0.5_fgsm_config.yaml test --model.adv_test_method=idAttack

python main.py --config config/pair_train_gamma_0.5_fgsm_config.yaml test \
                --model.adv_test_method=FGSMAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.eps=0.031

python main.py --config config/pair_train_gamma_0.5_fgsm_config.yaml test \
                --model.adv_test_method=PGDAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.alpha=0.031 \
                --model.adv_test_method.eps=0.008

python main.py --config config/pair_train_gamma_0.5_fgsm_config.yaml test \
                --model.adv_test_method=DeepfoolAttack \
                --model.adv_test_method.max_iters=20 \
                --model.adv_test_method.overshoot=0.02

# gamma=6
python main.py --config config/pair_train_fgsm_config.yaml fit

python main.py --config config/pair_train_fgsm_config.yaml test --model.adv_test_method=idAttack

python main.py --config config/pair_train_fgsm_config.yaml test \
                --model.adv_test_method=FGSMAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.eps=0.031

python main.py --config config/pair_train_fgsm_config.yaml test \
                --model.adv_test_method=PGDAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.alpha=0.031 \
                --model.adv_test_method.eps=0.008

python main.py --config config/pair_train_fgsm_config.yaml test \
                --model.adv_test_method=DeepfoolAttack \
                --model.adv_test_method.max_iters=20 \
                --model.adv_test_method.overshoot=0.02

# gamma=2
python main.py --config config/pair_train_gamma_1_fgsm_config.yaml fit 

python main.py --config config/pair_train_gamma_1_fgsm_config.yaml test --model.adv_test_method=idAttack

python main.py --config config/pair_train_gamma_1_fgsm_config.yaml test \
                --model.adv_test_method=FGSMAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.eps=0.031

python main.py --config config/pair_train_gamma_1_fgsm_config.yaml test \
                --model.adv_test_method=PGDAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.alpha=0.031 \
                --model.adv_test_method.eps=0.008

python main.py --config config/pair_train_gamma_1_fgsm_config.yaml test \
                --model.adv_test_method=DeepfoolAttack \
                --model.adv_test_method.max_iters=20 \
                --model.adv_test_method.overshoot=0.02

# gamma=3
python main.py --config config/pair_train_gamma_3_fgsm_config.yaml fit

python main.py --config config/pair_train_gamma_3_fgsm_config.yaml test --model.adv_test_method=idAttack

python main.py --config config/pair_train_gamma_3_fgsm_config.yaml test \
                --model.adv_test_method=FGSMAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.eps=0.031

python main.py --config config/pair_train_gamma_3_fgsm_config.yaml test \
                --model.adv_test_method=PGDAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.alpha=0.031 \
                --model.adv_test_method.eps=0.008

python main.py --config config/pair_train_gamma_3_fgsm_config.yaml test \
                --model.adv_test_method=DeepfoolAttack \
                --model.adv_test_method.max_iters=20 \
                --model.adv_test_method.overshoot=0.02

# gamma=7
python main.py --config config/pair_train_gamma_7_fgsm_config.yaml fit

python main.py --config config/pair_train_gamma_7_fgsm_config.yaml test --model.adv_test_method=idAttack

python main.py --config config/pair_train_gamma_7_fgsm_config.yaml test \
                --model.adv_test_method=FGSMAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.eps=0.031

python main.py --config config/pair_train_gamma_7_fgsm_config.yaml test \
                --model.adv_test_method=PGDAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.alpha=0.031 \
                --model.adv_test_method.eps=0.008

python main.py --config config/pair_train_gamma_7_fgsm_config.yaml test \
                --model.adv_test_method=DeepfoolAttack \
                --model.adv_test_method.max_iters=20 \
                --model.adv_test_method.overshoot=0.02