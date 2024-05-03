# Embedding 1
python main.py --config config/pair_train_embed_1_fgsm_config.yaml fit

python main.py --config config/pair_train_embed_1_fgsm_config.yaml test --model.adv_test_method=idAttack

python main.py --config config/pair_train_embed_1_fgsm_config.yaml test \
                --model.adv_test_method=FGSMAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.eps=0.031

python main.py --config config/pair_train_embed_1_fgsm_config.yaml test \
                --model.adv_test_method=PGDAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.alpha=0.031 \
                --model.adv_test_method.eps=0.008

python main.py --config config/pair_train_embed_1_fgsm_config.yaml test \
                --model.adv_test_method=DeepfoolAttack \
                --model.adv_test_method.max_iters=20 \
                --model.adv_test_method.overshoot=0.02

# Embedding 2
python main.py --config config/pair_train_embed_2_fgsm_config.yaml fit

python main.py --config config/pair_train_embed_2_fgsm_config.yaml test --model.adv_test_method=idAttack

python main.py --config config/pair_train_embed_2_fgsm_config.yaml test \
                --model.adv_test_method=FGSMAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.eps=0.031

python main.py --config config/pair_train_embed_2_fgsm_config.yaml test \
                --model.adv_test_method=PGDAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.alpha=0.031 \
                --model.adv_test_method.eps=0.008

python main.py --config config/pair_train_embed_2_fgsm_config.yaml test \
                --model.adv_test_method=DeepfoolAttack \
                --model.adv_test_method.max_iters=20 \
                --model.adv_test_method.overshoot=0.02

# Embedding 3
python main.py --config config/pair_train_embed_3_fgsm_config.yaml fit

python main.py --config config/pair_train_embed_3_fgsm_config.yaml test --model.adv_test_method=idAttack

python main.py --config config/pair_train_embed_3_fgsm_config.yaml test \
                --model.adv_test_method=FGSMAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.eps=0.031

python main.py --config config/pair_train_embed_3_fgsm_config.yaml test \
                --model.adv_test_method=PGDAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.alpha=0.031 \
                --model.adv_test_method.eps=0.008

python main.py --config config/pair_train_embed_3_fgsm_config.yaml test \
                --model.adv_test_method=DeepfoolAttack \
                --model.adv_test_method.max_iters=20 \
                --model.adv_test_method.overshoot=0.02

# Embedding 4
python main.py --config config/pair_train_embed_4_fgsm_config.yaml fit 

python main.py --config config/pair_train_embed_4_fgsm_config.yaml test --model.adv_test_method=idAttack

python main.py --config config/pair_train_embed_4_fgsm_config.yaml test \
                --model.adv_test_method=FGSMAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.eps=0.031

python main.py --config config/pair_train_embed_4_fgsm_config.yaml test \
                --model.adv_test_method=PGDAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.alpha=0.031 \
                --model.adv_test_method.eps=0.008

python main.py --config config/pair_train_embed_4_fgsm_config.yaml test \
                --model.adv_test_method=DeepfoolAttack \
                --model.adv_test_method.max_iters=20 \
                --model.adv_test_method.overshoot=0.02

# Embedding 6
python main.py --config config/pair_train_embed_6_fgsm_config.yaml fit

python main.py --config config/pair_train_embed_6_fgsm_config.yaml test --model.adv_test_method=idAttack

python main.py --config config/pair_train_embed_6_fgsm_config.yaml test \
                --model.adv_test_method=FGSMAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.eps=0.031

python main.py --config config/pair_train_embed_6_fgsm_config.yaml test \
                --model.adv_test_method=PGDAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.alpha=0.031 \
                --model.adv_test_method.eps=0.008

python main.py --config config/pair_train_embed_6_fgsm_config.yaml test \
                --model.adv_test_method=DeepfoolAttack \
                --model.adv_test_method.max_iters=20 \
                --model.adv_test_method.overshoot=0.02