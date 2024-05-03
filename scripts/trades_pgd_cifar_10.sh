python main.py --config config/trades_pgd_config.yaml fit

python main.py --config config/trades_pgd_config.yaml test --model.adv_test_method=idAttack

python main.py --config config/trades_pgd_config.yaml test \
                --model.adv_test_method=FGSMAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.eps=0.031

python main.py --config config/trades_pgd_config.yaml test \
                --model.adv_test_method=PGDAttack \
                --model.adv_test_method.loss_module=CrossEntropyLoss \
                --model.adv_test_method.alpha=0.031 \
                --model.adv_test_method.eps=0.008

python main.py --config config/trades_pgd_config.yaml test \
                --model.adv_test_method=DeepfoolAttack \
                --model.adv_test_method.max_iters=20 \
                --model.adv_test_method.overshoot=0.02
