rm -r ./models/
rm -r ./lightning_logs/
mv ./results.csv ./results_last_experiments.csv

sh scripts/baseline_mnist.sh
sh scripts/adv_train_fgsm_mnist.sh
sh scripts/adv_train_pgd_mnist.sh
sh scripts/trades_fgsm_mnist.sh
sh scripts/trades_pgd_mnist.sh
sh scripts/pair_train_fgsm_mnist.sh
sh scripts/pair_train_pgd_mnist.sh
