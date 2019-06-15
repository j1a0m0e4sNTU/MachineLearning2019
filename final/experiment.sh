# python3 main.py train -info "0608 | Test -- MLP(200, 50, 3) for first 200 feature" -record records/0608_test.txt
# python3 main.py train -lr 0.0001 -info "0608 | MLP (C) (200, 600, 3) for first 200 feature" -record records/0608_01.txt
# python3 main.py train -epoch 100 -lr 0.0001 -info "0608 | MLP (C) for first 200 feature" -record records/0608_02.txt
# python3 main.py train -epoch 100 -lr 0.001 -info "0608 | MLP (Base) for first 200 feature" -record records/0608_03.txt
# python3 main.py train -epoch 100 -lr 0.001 -info "0608 | Pure linear model for first 200 feature" -record records/0608_purelinear.txt
# python3 main.py train -epoch 100 -lr 0.001 -info "0608 | Pure linear model + BN + ReLU for first 200 feature" -record records/0608_linear_BN_ReLU.txt
# python3 main.py train -epoch 100 -lr 0.001 -info "0608 | MLP (A) for first 200 feature" -record records/0608_mlp_A.txt
# python3 main.py train -epoch 100 -lr 0.001 -info "0608 | MLP (B) for first 200 feature" -record records/0608_mlp_B.txt
# python3 main.py train -epoch 100 -lr 0.001 -info "0608 | MLP (C) for first 200 feature" -record records/0608_mlp_C.txt
# python3 main.py train -input_dim 400 -model B -epoch 100 -lr 0.001 -info "0614 | MLP (B) | first 200 feature + quadratic term" -record records/0614_mlp_B.txt
# python3 main.py train -input_dim 400 -model C -epoch 100 -lr 0.001 -info "0614 | MLP (C) | first 200 feature + quadratic term" -record records/0614_mlp_C.txt
# python3 main.py train -input_dim 400 -model C -info "0614 | MLP (C) | first 200 feature + quadratic term | WMAELoss" -record records/0614_c_wmae.txt
# python3 main.py train -input_dim 400 -model C -info "0614 | MLP (C) | first 200 feature + quadratic term | NAELoss" -record records/0614_c_NAE.txt
# python3 main.py train -input_dim 400 -model C -info "0614 | MLP (C) | first 200 feature + quadratic term | ABSLoss" -record records/0615_c_ABS.txt
# python3 main.py train -input_dim 400 -model C_d -info "0615 | MLP (C_d) | first 200 feature + quadratic term | ABSLoss" -record records/0615_Cd_ABS.txt
# python3 main.py train -input_dim 400 -model D -info "0615 | MLP (D) | first 200 feature + quadratic term | ABSLoss" -record records/0615_D_ABS.txt
# python3 main.py train -input_dim 400 -model E -info "0615 | MLP (E) | first 200 feature + quadratic term | ABSLoss" -record records/0615_E_ABS.txt
# python3 main.py train -input_dim 400 -model F -info "0615 | MLP (F) | first 200 feature + quadratic term | ABSLoss" -record records/0615_F_ABS.txt
# python3 main.py train -model A -bs 128 -info "0615 | RNN (A) | first 200 feature | ABSLoss" -record records/0615_rnn_A.txt
python3 main.py train -model A -bs 128 -info "0615 | RNN (A) | first 200 feature - reverse | ABSLoss" -record records/0615_rnn_A_reverse.txt
