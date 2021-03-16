rl_replicate_unique_wtdecay/exp_0.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 1729 --wt-decay 1e-05 --latent-wt-decay 1e-05 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward acc --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-1e-05_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-acc_s-1729_sel-1_se-last_warmup_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-1e-05 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-1729_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-1e-05/checkpoints/checkpoint_last_warmup.pth &
rl_replicate_unique_wtdecay/exp_10.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 3120 --wt-decay 0.0001 --latent-wt-decay 0.0001 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward count --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-0.0001_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-count_s-3120_sel-1_se-last_warmup_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-0.0001 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-3120_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-0.0001/checkpoints/checkpoint_last_warmup.pth &
rl_replicate_unique_wtdecay/exp_11.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 3120 --wt-decay 0.0001 --latent-wt-decay 0.0001 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward count --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-0.0001_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-count_s-3120_sel-1_se-best_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-0.0001 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-3120_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
rl_replicate_unique_wtdecay/exp_12.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 3120 --wt-decay 1e-05 --latent-wt-decay 1e-05 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward acc --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-1e-05_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-acc_s-3120_sel-1_se-last_warmup_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-1e-05 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-3120_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-1e-05/checkpoints/checkpoint_last_warmup.pth &
rl_replicate_unique_wtdecay/exp_13.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 3120 --wt-decay 1e-05 --latent-wt-decay 1e-05 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward acc --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-1e-05_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-acc_s-3120_sel-1_se-best_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-1e-05 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-3120_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-1e-05/checkpoints/checkpoint_best.pth &
rl_replicate_unique_wtdecay/exp_14.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 3120 --wt-decay 1e-05 --latent-wt-decay 1e-05 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward count --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-1e-05_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-count_s-3120_sel-1_se-last_warmup_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-1e-05 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-3120_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-1e-05/checkpoints/checkpoint_last_warmup.pth &
rl_replicate_unique_wtdecay/exp_15.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 3120 --wt-decay 1e-05 --latent-wt-decay 1e-05 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward count --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-1e-05_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-count_s-3120_sel-1_se-best_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-1e-05 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-3120_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-1e-05/checkpoints/checkpoint_best.pth &
rl_replicate_unique_wtdecay/exp_1.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 1729 --wt-decay 1e-05 --latent-wt-decay 1e-05 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward acc --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-1e-05_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-acc_s-1729_sel-1_se-best_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-1e-05 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-1729_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-1e-05/checkpoints/checkpoint_best.pth &
rl_replicate_unique_wtdecay/exp_2.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 1729 --wt-decay 1e-05 --latent-wt-decay 1e-05 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward count --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-1e-05_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-count_s-1729_sel-1_se-last_warmup_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-1e-05 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-1729_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-1e-05/checkpoints/checkpoint_last_warmup.pth &
rl_replicate_unique_wtdecay/exp_3.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 1729 --wt-decay 1e-05 --latent-wt-decay 1e-05 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward count --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-1e-05_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-count_s-1729_sel-1_se-best_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-1e-05 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-1729_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-1e-05/checkpoints/checkpoint_best.pth &
rl_replicate_unique_wtdecay/exp_4.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 42 --wt-decay 1e-05 --latent-wt-decay 1e-05 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward acc --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-1e-05_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-acc_s-42_sel-1_se-last_warmup_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-1e-05 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-42_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-1e-05/checkpoints/checkpoint_last_warmup.pth &
rl_replicate_unique_wtdecay/exp_5.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 42 --wt-decay 1e-05 --latent-wt-decay 1e-05 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward acc --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-1e-05_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-acc_s-42_sel-1_se-best_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-1e-05 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-42_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-1e-05/checkpoints/checkpoint_best.pth &
rl_replicate_unique_wtdecay/exp_6.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 42 --wt-decay 1e-05 --latent-wt-decay 1e-05 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward count --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-1e-05_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-count_s-42_sel-1_se-last_warmup_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-1e-05 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-42_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-1e-05/checkpoints/checkpoint_last_warmup.pth &
rl_replicate_unique_wtdecay/exp_7.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 42 --wt-decay 1e-05 --latent-wt-decay 1e-05 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward count --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-1e-05_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-count_s-42_sel-1_se-best_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-1e-05 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-42_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-1e-05/checkpoints/checkpoint_best.pth &
rl_replicate_unique_wtdecay/exp_8.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 3120 --wt-decay 0.0001 --latent-wt-decay 0.0001 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward acc --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-0.0001_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-acc_s-3120_sel-1_se-last_warmup_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-0.0001 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-3120_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-0.0001/checkpoints/checkpoint_last_warmup.pth &
rl_replicate_unique_wtdecay/exp_9.sh:jac-run /home/cse/phd/csz178057/phd/neural-logic-machines/scripts/graph/exploration_replicate.py --task nqueens --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --seed 3120 --wt-decay 0.0001 --latent-wt-decay 0.0001 --train-number 10 --num-missing-queens 5 --epochs 200 --warmup-epochs 200 --nlm-depth 30 --test-number-begin 11 --test-number-end 11 --hot-data-sampling ambiguous --latent-reg-wt 1.0 --lr-hot 0.0005 --latent-margin-min 0.5 --latent-margin-fraction 1.414 --rl-reward acc --pretrain-phi 1 --latent-aux-loss rl --latent-aux-loss-factor 1 --selector-model 1 --latent-dis-prob softmax --warmup-data-sampling unique --min-loss 0 --no-static 0 --copy-back-frequency 10 --dump-dir models/rl_replicate_unique_wtdecay/cpf-10_e-200_hds-ambiguous_lal-rl_lalf-1_prob-softmax_lmf-1.414_lmm-0.5_lrw-1.0_ltwd-0.0001_lrh-0.0005_min-0_d-30_nos-0_nm-5_phip-1_rlrc-acc_s-3120_sel-1_se-best_tsb-11_tse-11_tr-10_wds-unique_we-200_wtd-0.0001 --train-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_data_10_5.pkl --test-file /home/cse/phd/csz178057/hpcscratch/nlm/data/nqueens_11_6_dev.pkl --load-checkpoint /home/cse/phd/csz178057/phd/neural-logic-machines/models/bl_replicate_unique_wtdecay/arb-0_e-0_lrh-0.0005_min-0_d-30_nm-5_s-3120_tesb-11_tese-11_trs-10_wds-unique_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
