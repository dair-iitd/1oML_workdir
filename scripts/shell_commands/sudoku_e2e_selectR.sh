jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling unique --min-loss 0 --lr-hot 2e-05 --latent-model mlp --latent-hidden-list 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 0 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-0_lal-rl_lalf-1_prob-softmax_lhid-32_lmod-mlp_lip-0_lwtd-0.0001_lrh-2e-05_lrl-0.0005_min-0_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-best_upgnorm-1000_wds-unique_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-0_s-3120_tesb-9_tese-9_trs-9_wds-unique_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling four-one --min-loss 1 --lr-hot 2e-05 --latent-model conv --latent-hidden-list 100 64 32 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 0 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-0_lal-rl_lalf-1_prob-softmax_lhid-100.64.32.32_lmod-conv_lip-0_lwtd-0.0001_lrh-2e-05_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-best_upgnorm-1000_wds-four-one_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-four-one_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling four-one --min-loss 1 --lr-hot 2e-05 --latent-model conv --latent-hidden-list 100 64 32 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 1 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-1_lal-rl_lalf-1_prob-softmax_lhid-100.64.32.32_lmod-conv_lip-0_lwtd-0.0001_lrh-2e-05_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-best_upgnorm-1000_wds-four-one_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-1_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-four-one_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling four-one --min-loss 1 --lr-hot 0.0001 --latent-model mlp --latent-hidden-list 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 0 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-0_lal-rl_lalf-1_prob-softmax_lhid-32_lmod-mlp_lip-0_lwtd-0.0001_lrh-0.0001_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-50_upgnorm-1000_wds-four-one_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-four-one_we-200_wtd-0.0001/checkpoints/checkpoint_50.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling four-one --min-loss 1 --lr-hot 0.0001 --latent-model mlp --latent-hidden-list 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 1 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-1_lal-rl_lalf-1_prob-softmax_lhid-32_lmod-mlp_lip-0_lwtd-0.0001_lrh-0.0001_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-50_upgnorm-1000_wds-four-one_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-1_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-four-one_we-200_wtd-0.0001/checkpoints/checkpoint_50.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling four-one --min-loss 1 --lr-hot 0.0001 --latent-model conv --latent-hidden-list 100 64 32 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 0 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-0_lal-rl_lalf-1_prob-softmax_lhid-100.64.32.32_lmod-conv_lip-0_lwtd-0.0001_lrh-0.0001_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-50_upgnorm-1000_wds-four-one_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-four-one_we-200_wtd-0.0001/checkpoints/checkpoint_50.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling four-one --min-loss 1 --lr-hot 0.0001 --latent-model conv --latent-hidden-list 100 64 32 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 1 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-1_lal-rl_lalf-1_prob-softmax_lhid-100.64.32.32_lmod-conv_lip-0_lwtd-0.0001_lrh-0.0001_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-50_upgnorm-1000_wds-four-one_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-1_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-four-one_we-200_wtd-0.0001/checkpoints/checkpoint_50.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling rs --min-loss 1 --lr-hot 2e-05 --latent-model mlp --latent-hidden-list 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 0 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-0_lal-rl_lalf-1_prob-softmax_lhid-32_lmod-mlp_lip-0_lwtd-0.0001_lrh-2e-05_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-best_upgnorm-1000_wds-rs_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-rs_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling rs --min-loss 1 --lr-hot 2e-05 --latent-model mlp --latent-hidden-list 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 1 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-1_lal-rl_lalf-1_prob-softmax_lhid-32_lmod-mlp_lip-0_lwtd-0.0001_lrh-2e-05_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-best_upgnorm-1000_wds-rs_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-1_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-rs_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling rs --min-loss 1 --lr-hot 2e-05 --latent-model conv --latent-hidden-list 100 64 32 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 0 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-0_lal-rl_lalf-1_prob-softmax_lhid-100.64.32.32_lmod-conv_lip-0_lwtd-0.0001_lrh-2e-05_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-best_upgnorm-1000_wds-rs_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-rs_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling rs --min-loss 1 --lr-hot 2e-05 --latent-model conv --latent-hidden-list 100 64 32 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 1 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-1_lal-rl_lalf-1_prob-softmax_lhid-100.64.32.32_lmod-conv_lip-0_lwtd-0.0001_lrh-2e-05_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-best_upgnorm-1000_wds-rs_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-1_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-rs_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling unique --min-loss 0 --lr-hot 2e-05 --latent-model mlp --latent-hidden-list 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 1 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-1_lal-rl_lalf-1_prob-softmax_lhid-32_lmod-mlp_lip-0_lwtd-0.0001_lrh-2e-05_lrl-0.0005_min-0_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-best_upgnorm-1000_wds-unique_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-0_s-3120_tesb-9_tese-9_trs-9_wds-unique_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling rs --min-loss 1 --lr-hot 0.0001 --latent-model mlp --latent-hidden-list 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 0 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-0_lal-rl_lalf-1_prob-softmax_lhid-32_lmod-mlp_lip-0_lwtd-0.0001_lrh-0.0001_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-50_upgnorm-1000_wds-rs_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-rs_we-200_wtd-0.0001/checkpoints/checkpoint_50.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling rs --min-loss 1 --lr-hot 0.0001 --latent-model mlp --latent-hidden-list 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 1 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-1_lal-rl_lalf-1_prob-softmax_lhid-32_lmod-mlp_lip-0_lwtd-0.0001_lrh-0.0001_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-50_upgnorm-1000_wds-rs_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-1_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-rs_we-200_wtd-0.0001/checkpoints/checkpoint_50.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling rs --min-loss 1 --lr-hot 0.0001 --latent-model conv --latent-hidden-list 100 64 32 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 0 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-0_lal-rl_lalf-1_prob-softmax_lhid-100.64.32.32_lmod-conv_lip-0_lwtd-0.0001_lrh-0.0001_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-50_upgnorm-1000_wds-rs_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-rs_we-200_wtd-0.0001/checkpoints/checkpoint_50.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling rs --min-loss 1 --lr-hot 0.0001 --latent-model conv --latent-hidden-list 100 64 32 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 1 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-1_lal-rl_lalf-1_prob-softmax_lhid-100.64.32.32_lmod-conv_lip-0_lwtd-0.0001_lrh-0.0001_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-50_upgnorm-1000_wds-rs_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-1_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-rs_we-200_wtd-0.0001/checkpoints/checkpoint_50.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling unique --min-loss 0 --lr-hot 2e-05 --latent-model conv --latent-hidden-list 100 64 32 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 0 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-0_lal-rl_lalf-1_prob-softmax_lhid-100.64.32.32_lmod-conv_lip-0_lwtd-0.0001_lrh-2e-05_lrl-0.0005_min-0_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-best_upgnorm-1000_wds-unique_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-0_s-3120_tesb-9_tese-9_trs-9_wds-unique_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling unique --min-loss 0 --lr-hot 2e-05 --latent-model conv --latent-hidden-list 100 64 32 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 1 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-1_lal-rl_lalf-1_prob-softmax_lhid-100.64.32.32_lmod-conv_lip-0_lwtd-0.0001_lrh-2e-05_lrl-0.0005_min-0_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-best_upgnorm-1000_wds-unique_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-0_s-3120_tesb-9_tese-9_trs-9_wds-unique_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling unique --min-loss 0 --lr-hot 0.0001 --latent-model mlp --latent-hidden-list 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 0 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-0_lal-rl_lalf-1_prob-softmax_lhid-32_lmod-mlp_lip-0_lwtd-0.0001_lrh-0.0001_lrl-0.0005_min-0_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-50_upgnorm-1000_wds-unique_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-0_s-3120_tesb-9_tese-9_trs-9_wds-unique_we-200_wtd-0.0001/checkpoints/checkpoint_50.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling unique --min-loss 0 --lr-hot 0.0001 --latent-model mlp --latent-hidden-list 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 1 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-1_lal-rl_lalf-1_prob-softmax_lhid-32_lmod-mlp_lip-0_lwtd-0.0001_lrh-0.0001_lrl-0.0005_min-0_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-50_upgnorm-1000_wds-unique_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-0_s-3120_tesb-9_tese-9_trs-9_wds-unique_we-200_wtd-0.0001/checkpoints/checkpoint_50.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling unique --min-loss 0 --lr-hot 0.0001 --latent-model conv --latent-hidden-list 100 64 32 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 0 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-0_lal-rl_lalf-1_prob-softmax_lhid-100.64.32.32_lmod-conv_lip-0_lwtd-0.0001_lrh-0.0001_lrl-0.0005_min-0_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-50_upgnorm-1000_wds-unique_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-0_s-3120_tesb-9_tese-9_trs-9_wds-unique_we-200_wtd-0.0001/checkpoints/checkpoint_50.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling unique --min-loss 0 --lr-hot 0.0001 --latent-model conv --latent-hidden-list 100 64 32 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 1 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-1_lal-rl_lalf-1_prob-softmax_lhid-100.64.32.32_lmod-conv_lip-0_lwtd-0.0001_lrh-0.0001_lrl-0.0005_min-0_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-50_upgnorm-1000_wds-unique_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-0_s-3120_tesb-9_tese-9_trs-9_wds-unique_we-200_wtd-0.0001/checkpoints/checkpoint_50.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling four-one --min-loss 1 --lr-hot 2e-05 --latent-model mlp --latent-hidden-list 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 0 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-0_lal-rl_lalf-1_prob-softmax_lhid-32_lmod-mlp_lip-0_lwtd-0.0001_lrh-2e-05_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-best_upgnorm-1000_wds-four-one_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-0_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-four-one_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
jac-run scripts/graph/exploration.py --task sudoku --save-interval 10 --nlm-residual 1 --use-gpu --latent-residual 1 --skip-warmup 1 --test-interval 1 --model rrn --batch-size 16 --test-batch-size 32 --epoch-size 625 --train-number 9 --test-number-begin 9 --test-number-end 9 --latent-reg-wt 1.0 --latent-margin-fraction 1.414 --warmup-data-sampling four-one --min-loss 1 --lr-hot 2e-05 --latent-model mlp --latent-hidden-list 32 --epochs 200 --warmup-epochs 200 --seed 3120 --hot-data-sampling rs --rl-reward count --latent-dis-prob softmax --pretrain-phi 2 --selector-model 1 --copy-back-frequency 0 --no-static 1 --lr-latent 0.0005 --latent-aux-loss rl --latent-aux-loss-factor 1 --grad-clip 5.0 --latent-sudoku-input-prob 0 --wt-decay 0.0001 --latent-wt-decay 0.0001 --upper-limit-on-grad-norm 1000 --incomplete-targetset 1 --dump-dir sudoku_models/rl_replicate/cpf-0_e-200_clip-5.0_hds-rs_add-1_lal-rl_lalf-1_prob-softmax_lhid-32_lmod-mlp_lip-0_lwtd-0.0001_lrh-2e-05_lrl-0.0005_min-1_nos-1_phip-2_rlrc-count_s-3120_sel-1_se-best_upgnorm-1000_wds-four-one_we-200_wtd-0.0001 --train-file data/sudoku_9_train_e.pkl --test-file data/sudoku_9_dev_e.pkl --load-checkpoint sudoku_models/bl_replicate/arb-0_e-0_clip-5.0_add-1_lr-0.001_min-1_s-3120_tesb-9_tese-9_trs-9_wds-four-one_we-200_wtd-0.0001/checkpoints/checkpoint_best.pth &
