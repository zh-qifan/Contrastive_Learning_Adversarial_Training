rm -r ./models/
rm -r ./lightning_logs/
mv ./results.csv ./results_last_experiments.csv

sh scripts/baseline_cifar_10.sh
sh scripts/adv_train_fgsm_cifar_10.sh
sh scripts/adv_train_pgd_cifar_10.sh
sh scripts/trades_fgsm_cifar_10.sh
sh scripts/trades_pgd_cifar_10.sh
sh scripts/pair_train_fgsm_cifar_10.sh
sh scripts/pair_train_pgd_cifar_10.sh
