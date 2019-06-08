# python3 main.py train -info "0608 | Test -- MLP(200, 50, 3) for first 200 feature" -record records/0608_test.txt
# python3 main.py train -lr 0.0001 -info "0608 | MLP (C) (200, 600, 3) for first 200 feature" -record records/0608_01.txt
# python3 main.py train -epoch 100 -lr 0.0001 -info "0608 | MLP (C) for first 200 feature" -record records/0608_02.txt
# python3 main.py train -epoch 100 -lr 0.001 -info "0608 | MLP (Base) for first 200 feature" -record records/0608_03.txt
# python3 main.py train -epoch 100 -lr 0.001 -info "0608 | Pure linear model for first 200 feature" -record records/0608_purelinear.txt
python3 main.py train -epoch 100 -lr 0.001 -info "0608 | Pure linear model + BN for first 200 feature" -record records/0608_linear_BN.txt
