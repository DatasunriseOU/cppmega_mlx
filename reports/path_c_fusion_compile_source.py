@T.prim_func
def local_gb10_quarter_path_c_10_12(
    path_c_float32_abi_bank: T.Buffer((385269133,), "float32"),
    path_c_uint8_abi_bank: T.Buffer((14680064,), "uint8"),
    path_c_int32_abi_bank: T.Buffer((1,), "int32"),
    local_gb10_quarter_brick_10_M_mamba3_conv_history: T.Buffer((2, 11264), "float32"),
    local_gb10_quarter_brick_10_M_mamba3_projected_vec: T.Buffer((18784,), "float32"),
    local_gb10_quarter_brick_10_M_mamba3_conv_vec: T.Buffer((11264,), "float32"),
    local_gb10_quarter_brick_10_M_mamba3_out_inner: T.Buffer((7168,), "float32"),
    local_gb10_quarter_brick_11_R_m2rnn_h_state: T.Buffer((4, 64, 16), "float32"),
    local_gb10_quarter_brick_11_R_m2rnn_h_next: T.Buffer((4, 64, 16), "float32"),
    local_gb10_quarter_brick_10_M_delta: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_10_M_hidden_after: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_11_R_residual_norm_hidden: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_11_R_delta: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_12_A_residual_norm_hidden: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_12_A_qkv_projection_q_fp8_grad: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_12_A_qkv_projection_kv_fp8_grad: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_12_A_residual_norm_hidden_grad: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_10_M_hidden_after_grad: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_11_R_delta_grad: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_11_R_residual_norm_hidden_grad: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_10_M_delta_grad: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_context_values: T.Buffer((3584,), "float32"),
    local_gb10_quarter_brick_10_M_mamba3_angle_cumsum: T.Buffer((112, 16), "float32"),
):
    with T.Kernel(1, threads=256):
        # internal_buffer_policy: row_local_hidden
        # loop_policy: row_phased_hidden
        lane = T.get_thread_binding(0)
        local_gb10_quarter_brick_12_A_qkv_projection_q_fp8 = T.alloc_shared((3584,), "uint8")
        local_gb10_quarter_brick_12_A_qkv_projection_q_scale = T.alloc_shared((28,), "float32")
        local_gb10_quarter_brick_12_A_qkv_projection_indices = T.alloc_shared((448,), "int32")
        local_gb10_quarter_brick_12_A_qkv_projection_q_scale_grad = T.alloc_shared((28,), "float32")
        local_gb10_quarter_brick_12_A_qkv_projection_kv_scale_grad = T.alloc_shared((28,), "float32")
        local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_pair = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_pair = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_qkv_projection_attention_q_prepared = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_prepared = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sink_enabled = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index = T.alloc_local((1,), "int32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_max = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_weight = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sumexp = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_value_accum = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_context_accum = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_q_head = T.alloc_local((1,), "int32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_kv_head = T.alloc_local((1,), "int32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head = T.alloc_local((1,), "int32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_dim = T.alloc_local((1,), "int32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd_apply_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd_q_value = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd_kv_value = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_q_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_kv_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_rope_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_bwd_m2rnn_project_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_bwd_m2rnn_conv_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_bwd_m2rnn_recurrent_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_bwd_m2rnn_post_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_10_M_bwd_mamba3_project_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_10_M_bwd_mamba3_conv_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_10_M_bwd_mamba3_dt_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_10_M_bwd_mamba3_state_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_10_M_bwd_mamba3_out_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_residual_norm_row_sum_sq_partial = T.alloc_shared((256,), "float32")
        local_gb10_quarter_brick_11_R_residual_norm_row_sum_sq = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_residual_norm_row_inv_rms = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_residual_norm_row_sum_sq_partial = T.alloc_shared((256,), "float32")
        local_gb10_quarter_brick_12_A_residual_norm_row_sum_sq = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_residual_norm_row_inv_rms = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_10_M_mamba3_b_inv_rms = T.alloc_shared((4, 8), "float32")
        local_gb10_quarter_brick_10_M_mamba3_c_inv_rms = T.alloc_shared((4, 8), "float32")
        local_gb10_quarter_brick_10_M_mamba3_b_raw = T.alloc_shared((8, 64), "float32")
        local_gb10_quarter_brick_10_M_mamba3_c_raw = T.alloc_shared((8, 64), "float32")
        local_gb10_quarter_brick_10_M_mamba3_b_group = T.alloc_shared((8, 64), "float32")
        local_gb10_quarter_brick_10_M_mamba3_c_group = T.alloc_shared((8, 64), "float32")
        local_gb10_quarter_brick_10_M_mamba3_dt_vec = T.alloc_shared((112,), "float32")
        local_gb10_quarter_brick_10_M_mamba3_a_vec = T.alloc_shared((112,), "float32")
        local_gb10_quarter_brick_10_M_mamba3_trap_group = T.alloc_shared((8,), "float32")
        local_gb10_quarter_brick_10_M_mamba3_next_dt = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_10_M_mamba3_next_trap = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_10_M_mamba3_accum = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_10_M_mamba3_state_value = T.alloc_local((1,), "float32")
        # local_gb10_quarter_brick_10_M: mamba3_state_policy: external_scan_state
        for local_gb10_quarter_brick_10_M_angle_flat_init in T.serial(lane, 1792, step=256):
            local_gb10_quarter_brick_10_M_head_init = local_gb10_quarter_brick_10_M_angle_flat_init // 16
            local_gb10_quarter_brick_10_M_angle_init = local_gb10_quarter_brick_10_M_angle_flat_init % 16
            local_gb10_quarter_brick_10_M_mamba3_angle_cumsum[local_gb10_quarter_brick_10_M_head_init, local_gb10_quarter_brick_10_M_angle_init] = 0.0
        for local_gb10_quarter_brick_10_M_state_flat_init in T.serial(lane, 458752, step=256):
            local_gb10_quarter_brick_10_M_head_init = local_gb10_quarter_brick_10_M_state_flat_init // 4096
            local_gb10_quarter_brick_10_M_dim_init = (local_gb10_quarter_brick_10_M_state_flat_init // 64) % 64
            local_gb10_quarter_brick_10_M_state_idx_init = local_gb10_quarter_brick_10_M_state_flat_init % 64
            path_c_float32_abi_bank[108663008 + ((local_gb10_quarter_brick_10_M_head_init * 4096 + local_gb10_quarter_brick_10_M_dim_init * 64 + local_gb10_quarter_brick_10_M_state_idx_init) % 458752)] = path_c_float32_abi_bank[108204256 + ((local_gb10_quarter_brick_10_M_head_init * 4096 + local_gb10_quarter_brick_10_M_dim_init * 64 + local_gb10_quarter_brick_10_M_state_idx_init) % 458752)]
        T.sync_threads()
        # local_gb10_quarter_brick_10_M: mamba3_conv_policy: zero_padded_ring_history
        for local_gb10_quarter_brick_10_M_history_flat_init in T.serial(lane, 22528, step=256):
            local_gb10_quarter_brick_10_M_hist_init = local_gb10_quarter_brick_10_M_history_flat_init // 11264
            local_gb10_quarter_brick_10_M_conv_ch_init = local_gb10_quarter_brick_10_M_history_flat_init % 11264
            local_gb10_quarter_brick_10_M_mamba3_conv_history[local_gb10_quarter_brick_10_M_hist_init, local_gb10_quarter_brick_10_M_conv_ch_init] = 0.0
        T.sync_threads()
        local_gb10_quarter_brick_11_R_m2rnn_projected_vec = T.alloc_shared((226,), "float32")
        local_gb10_quarter_brick_11_R_m2rnn_conv_vec = T.alloc_shared((160,), "float32")
        local_gb10_quarter_brick_11_R_m2rnn_post_vec = T.alloc_shared((64,), "float32")
        local_gb10_quarter_brick_11_R_m2rnn_conv_history = T.alloc_shared((3, 160), "float32")
        local_gb10_quarter_brick_11_R_m2rnn_accum = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_m2rnn_decay = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_m2rnn_sum_sq = T.alloc_shared((1,), "float32")
        local_gb10_quarter_brick_11_R_m2rnn_sum_sq_partial = T.alloc_shared((256,), "float32")
        local_gb10_quarter_brick_11_R_m2rnn_inv_rms = T.alloc_shared((1,), "float32")
        # local_gb10_quarter_brick_11_R: m2rnn_state_policy: row_carried
        for local_gb10_quarter_brick_11_R_state_idx_init in T.serial(lane, 4096, step=256):
            local_gb10_quarter_brick_11_R_head_init = local_gb10_quarter_brick_11_R_state_idx_init // 1024
            local_gb10_quarter_brick_11_R_kk_init = (local_gb10_quarter_brick_11_R_state_idx_init // 16) % 64
            local_gb10_quarter_brick_11_R_vv_init = local_gb10_quarter_brick_11_R_state_idx_init % 16
            local_gb10_quarter_brick_11_R_m2rnn_h_state[local_gb10_quarter_brick_11_R_head_init, local_gb10_quarter_brick_11_R_kk_init, local_gb10_quarter_brick_11_R_vv_init] = path_c_float32_abi_bank[124845960 + ((local_gb10_quarter_brick_11_R_head_init * 1024 + local_gb10_quarter_brick_11_R_kk_init * 16 + local_gb10_quarter_brick_11_R_vv_init) % 4096)]
            local_gb10_quarter_brick_11_R_m2rnn_h_next[local_gb10_quarter_brick_11_R_head_init, local_gb10_quarter_brick_11_R_kk_init, local_gb10_quarter_brick_11_R_vv_init] = path_c_float32_abi_bank[124845960 + ((local_gb10_quarter_brick_11_R_head_init * 1024 + local_gb10_quarter_brick_11_R_kk_init * 16 + local_gb10_quarter_brick_11_R_vv_init) % 4096)]
        T.sync_threads()
        # local_gb10_quarter_brick_11_R: m2rnn_conv_policy: ring_history
        for local_gb10_quarter_brick_11_R_history_idx_init in T.serial(lane, 480, step=256):
            local_gb10_quarter_brick_11_R_hist_init = local_gb10_quarter_brick_11_R_history_idx_init // 160
            local_gb10_quarter_brick_11_R_conv_ch_init = local_gb10_quarter_brick_11_R_history_idx_init % 160
            local_gb10_quarter_brick_11_R_m2rnn_conv_history[local_gb10_quarter_brick_11_R_hist_init, local_gb10_quarter_brick_11_R_conv_ch_init] = path_c_float32_abi_bank[124850056 + ((local_gb10_quarter_brick_11_R_hist_init * 160 + local_gb10_quarter_brick_11_R_conv_ch_init) % 480)]
        T.sync_threads()
        local_gb10_quarter_brick_12_A_residual_norm_bwd_row_sum_sq_partial = T.alloc_shared((256,), "float32")
        local_gb10_quarter_brick_12_A_residual_norm_bwd_row_dot_partial = T.alloc_shared((256,), "float32")
        local_gb10_quarter_brick_12_A_residual_norm_bwd_row_sum_sq = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_residual_norm_bwd_row_inv_rms = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_residual_norm_bwd_row_dot = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_residual_norm_bwd_row_norm_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_12_A_residual_norm_bwd_row_total_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_residual_norm_bwd_row_sum_sq_partial = T.alloc_shared((256,), "float32")
        local_gb10_quarter_brick_11_R_residual_norm_bwd_row_dot_partial = T.alloc_shared((256,), "float32")
        local_gb10_quarter_brick_11_R_residual_norm_bwd_row_sum_sq = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_residual_norm_bwd_row_inv_rms = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_residual_norm_bwd_row_dot = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_residual_norm_bwd_row_norm_grad = T.alloc_local((1,), "float32")
        local_gb10_quarter_brick_11_R_residual_norm_bwd_row_total_grad = T.alloc_local((1,), "float32")
        for row in T.serial(0, 4096):
            # local_gb10_quarter_brick_10_M: mamba3_mimo
            # local_gb10_quarter_brick_10_M production_fragment_status: production_region_inlined
            # local_gb10_quarter_brick_10_M production_fragment_reason: row-phased descriptor codegen fuses Mamba3 dense input projection, causal depthwise convolution, B/C norm+RoPE, scan-state recurrence, gate, and output projection from the block-level ABI without full activation staging
            # mamba3_projection_policy: dense_row_local
            for local_gb10_quarter_brick_10_M_proj_dim in T.serial(lane, 18784, step=256):
                local_gb10_quarter_brick_10_M_mamba3_projected_vec[local_gb10_quarter_brick_10_M_proj_dim] = 0.0
                for local_gb10_quarter_brick_10_M_hidden_dim in T.serial(0, 3584):
                    local_gb10_quarter_brick_10_M_mamba3_projected_vec[local_gb10_quarter_brick_10_M_proj_dim] = local_gb10_quarter_brick_10_M_mamba3_projected_vec[local_gb10_quarter_brick_10_M_proj_dim] + (path_c_float32_abi_bank[row * 3584 + local_gb10_quarter_brick_10_M_hidden_dim] * path_c_float32_abi_bank[15138816 + ((local_gb10_quarter_brick_10_M_proj_dim * 3584 + local_gb10_quarter_brick_10_M_hidden_dim))])
            T.sync_threads()
            # mamba3_conv_policy: causal_depthwise_ring_history
            for local_gb10_quarter_brick_10_M_conv_ch in T.serial(lane, 11264, step=256):
                local_gb10_quarter_brick_10_M_mamba3_conv_vec[local_gb10_quarter_brick_10_M_conv_ch] = path_c_float32_abi_bank[108184576 + ((local_gb10_quarter_brick_10_M_conv_ch) % 11264)]
                for local_gb10_quarter_brick_10_M_kernel_pos in T.serial(0, 2):
                    local_gb10_quarter_brick_10_M_mamba3_conv_vec[local_gb10_quarter_brick_10_M_conv_ch] = local_gb10_quarter_brick_10_M_mamba3_conv_vec[local_gb10_quarter_brick_10_M_conv_ch] + (local_gb10_quarter_brick_10_M_mamba3_conv_history[local_gb10_quarter_brick_10_M_kernel_pos, local_gb10_quarter_brick_10_M_conv_ch] * path_c_float32_abi_bank[108150784 + ((local_gb10_quarter_brick_10_M_conv_ch * 3 + local_gb10_quarter_brick_10_M_kernel_pos) % 33792)])
                local_gb10_quarter_brick_10_M_mamba3_conv_vec[local_gb10_quarter_brick_10_M_conv_ch] = local_gb10_quarter_brick_10_M_mamba3_conv_vec[local_gb10_quarter_brick_10_M_conv_ch] + (local_gb10_quarter_brick_10_M_mamba3_projected_vec[7168 + local_gb10_quarter_brick_10_M_conv_ch] * path_c_float32_abi_bank[108150784 + ((local_gb10_quarter_brick_10_M_conv_ch * 3 + 2) % 33792)])
                local_gb10_quarter_brick_10_M_mamba3_conv_vec[local_gb10_quarter_brick_10_M_conv_ch] = local_gb10_quarter_brick_10_M_mamba3_conv_vec[local_gb10_quarter_brick_10_M_conv_ch] * (1.0 / (1.0 + T.exp(-local_gb10_quarter_brick_10_M_mamba3_conv_vec[local_gb10_quarter_brick_10_M_conv_ch])))
            T.sync_threads()
            # mamba3_dt_policy: softplus_A_trapezoid
            for local_gb10_quarter_brick_10_M_head in T.serial(lane, 112, step=256):
                local_gb10_quarter_brick_10_M_mamba3_dt_vec[local_gb10_quarter_brick_10_M_head] = T.log(1.0 + T.exp(local_gb10_quarter_brick_10_M_mamba3_projected_vec[18432 + local_gb10_quarter_brick_10_M_head] + path_c_float32_abi_bank[108195840 + ((local_gb10_quarter_brick_10_M_head) % 112)]))
                local_gb10_quarter_brick_10_M_mamba3_a_vec[local_gb10_quarter_brick_10_M_head] = T.min(-T.log(1.0 + T.exp(local_gb10_quarter_brick_10_M_mamba3_projected_vec[18544 + local_gb10_quarter_brick_10_M_head])), -0.01)
                for local_gb10_quarter_brick_10_M_angle in T.serial(0, 16):
                    local_gb10_quarter_brick_10_M_mamba3_angle_cumsum[local_gb10_quarter_brick_10_M_head, local_gb10_quarter_brick_10_M_angle] = local_gb10_quarter_brick_10_M_mamba3_angle_cumsum[local_gb10_quarter_brick_10_M_head, local_gb10_quarter_brick_10_M_angle] + (local_gb10_quarter_brick_10_M_mamba3_projected_vec[18768 + local_gb10_quarter_brick_10_M_angle] * local_gb10_quarter_brick_10_M_mamba3_dt_vec[local_gb10_quarter_brick_10_M_head])
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_trap_group_loop in T.serial(lane, 8, step=256):
                local_gb10_quarter_brick_10_M_mamba3_trap_group[local_gb10_quarter_brick_10_M_trap_group_loop] = 0.0
                for local_gb10_quarter_brick_10_M_head in T.serial(0, 14):
                    local_gb10_quarter_brick_10_M_mamba3_accum[0] = local_gb10_quarter_brick_10_M_trap_group_loop * 14 + local_gb10_quarter_brick_10_M_head
                    local_gb10_quarter_brick_10_M_mamba3_next_dt[0] = 0.0
                    local_gb10_quarter_brick_10_M_mamba3_next_trap[0] = 0.0
                    if row + 1 < 4096:
                        for local_gb10_quarter_brick_10_M_hidden_dim in T.serial(0, 3584):
                            local_gb10_quarter_brick_10_M_mamba3_next_dt[0] = local_gb10_quarter_brick_10_M_mamba3_next_dt[0] + (path_c_float32_abi_bank[(row + 1) * 3584 + local_gb10_quarter_brick_10_M_hidden_dim] * path_c_float32_abi_bank[15138816 + (((18432 + T.cast(local_gb10_quarter_brick_10_M_mamba3_accum[0], "int32")) * 3584 + local_gb10_quarter_brick_10_M_hidden_dim))])
                            local_gb10_quarter_brick_10_M_mamba3_next_trap[0] = local_gb10_quarter_brick_10_M_mamba3_next_trap[0] + (path_c_float32_abi_bank[(row + 1) * 3584 + local_gb10_quarter_brick_10_M_hidden_dim] * path_c_float32_abi_bank[15138816 + (((18656 + T.cast(local_gb10_quarter_brick_10_M_mamba3_accum[0], "int32")) * 3584 + local_gb10_quarter_brick_10_M_hidden_dim))])
                        local_gb10_quarter_brick_10_M_mamba3_next_dt[0] = T.log(1.0 + T.exp(local_gb10_quarter_brick_10_M_mamba3_next_dt[0] + path_c_float32_abi_bank[108195840 + ((T.cast(local_gb10_quarter_brick_10_M_mamba3_accum[0], "int32")) % 112)]))
                    local_gb10_quarter_brick_10_M_mamba3_trap_group[local_gb10_quarter_brick_10_M_trap_group_loop] = local_gb10_quarter_brick_10_M_mamba3_trap_group[local_gb10_quarter_brick_10_M_trap_group_loop] + ((local_gb10_quarter_brick_10_M_mamba3_next_dt[0] * (1.0 - (1.0 / (1.0 + T.exp(-local_gb10_quarter_brick_10_M_mamba3_next_trap[0]))))) + (local_gb10_quarter_brick_10_M_mamba3_dt_vec[T.cast(local_gb10_quarter_brick_10_M_mamba3_accum[0], "int32")] * (1.0 / (1.0 + T.exp(-local_gb10_quarter_brick_10_M_mamba3_projected_vec[18656 + T.cast(local_gb10_quarter_brick_10_M_mamba3_accum[0], "int32")])))))
                local_gb10_quarter_brick_10_M_mamba3_trap_group[local_gb10_quarter_brick_10_M_trap_group_loop] = local_gb10_quarter_brick_10_M_mamba3_trap_group[local_gb10_quarter_brick_10_M_trap_group_loop] / 14.0
            T.sync_threads()
            # mamba3_bc_policy: rank_group_rmsnorm_rope
            for local_gb10_quarter_brick_10_M_rank_group_flat in T.serial(lane, 32, step=256):
                local_gb10_quarter_brick_10_M_mamba3_b_inv_rms[(local_gb10_quarter_brick_10_M_rank_group_flat // 8), (local_gb10_quarter_brick_10_M_rank_group_flat % 8)] = 0.0
                local_gb10_quarter_brick_10_M_mamba3_c_inv_rms[(local_gb10_quarter_brick_10_M_rank_group_flat // 8), (local_gb10_quarter_brick_10_M_rank_group_flat % 8)] = 0.0
                for local_gb10_quarter_brick_10_M_state_idx in T.serial(0, 64):
                    local_gb10_quarter_brick_10_M_mamba3_b_inv_rms[(local_gb10_quarter_brick_10_M_rank_group_flat // 8), (local_gb10_quarter_brick_10_M_rank_group_flat % 8)] = local_gb10_quarter_brick_10_M_mamba3_b_inv_rms[(local_gb10_quarter_brick_10_M_rank_group_flat // 8), (local_gb10_quarter_brick_10_M_rank_group_flat % 8)] + (local_gb10_quarter_brick_10_M_mamba3_conv_vec[7168 + (((local_gb10_quarter_brick_10_M_rank_group_flat // 8) * 8 + (local_gb10_quarter_brick_10_M_rank_group_flat % 8)) * 64 + local_gb10_quarter_brick_10_M_state_idx)] * local_gb10_quarter_brick_10_M_mamba3_conv_vec[7168 + (((local_gb10_quarter_brick_10_M_rank_group_flat // 8) * 8 + (local_gb10_quarter_brick_10_M_rank_group_flat % 8)) * 64 + local_gb10_quarter_brick_10_M_state_idx)])
                    local_gb10_quarter_brick_10_M_mamba3_c_inv_rms[(local_gb10_quarter_brick_10_M_rank_group_flat // 8), (local_gb10_quarter_brick_10_M_rank_group_flat % 8)] = local_gb10_quarter_brick_10_M_mamba3_c_inv_rms[(local_gb10_quarter_brick_10_M_rank_group_flat // 8), (local_gb10_quarter_brick_10_M_rank_group_flat % 8)] + (local_gb10_quarter_brick_10_M_mamba3_conv_vec[9216 + (((local_gb10_quarter_brick_10_M_rank_group_flat // 8) * 8 + (local_gb10_quarter_brick_10_M_rank_group_flat % 8)) * 64 + local_gb10_quarter_brick_10_M_state_idx)] * local_gb10_quarter_brick_10_M_mamba3_conv_vec[9216 + (((local_gb10_quarter_brick_10_M_rank_group_flat // 8) * 8 + (local_gb10_quarter_brick_10_M_rank_group_flat % 8)) * 64 + local_gb10_quarter_brick_10_M_state_idx)])
                local_gb10_quarter_brick_10_M_mamba3_b_inv_rms[(local_gb10_quarter_brick_10_M_rank_group_flat // 8), (local_gb10_quarter_brick_10_M_rank_group_flat % 8)] = T.rsqrt((local_gb10_quarter_brick_10_M_mamba3_b_inv_rms[(local_gb10_quarter_brick_10_M_rank_group_flat // 8), (local_gb10_quarter_brick_10_M_rank_group_flat % 8)] / 64.0) + 0.00001)
                local_gb10_quarter_brick_10_M_mamba3_c_inv_rms[(local_gb10_quarter_brick_10_M_rank_group_flat // 8), (local_gb10_quarter_brick_10_M_rank_group_flat % 8)] = T.rsqrt((local_gb10_quarter_brick_10_M_mamba3_c_inv_rms[(local_gb10_quarter_brick_10_M_rank_group_flat // 8), (local_gb10_quarter_brick_10_M_rank_group_flat % 8)] / 64.0) + 0.00001)
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_group_state_flat in T.serial(lane, 512, step=256):
                local_gb10_quarter_brick_10_M_mamba3_b_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] = 0.0
                local_gb10_quarter_brick_10_M_mamba3_c_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] = 0.0
                for local_gb10_quarter_brick_10_M_rank in T.serial(0, 4):
                    local_gb10_quarter_brick_10_M_mamba3_b_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] = local_gb10_quarter_brick_10_M_mamba3_b_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] + ((local_gb10_quarter_brick_10_M_mamba3_conv_vec[7168 + ((local_gb10_quarter_brick_10_M_rank * 8 + (local_gb10_quarter_brick_10_M_group_state_flat // 64)) * 64 + (local_gb10_quarter_brick_10_M_group_state_flat % 64))] * local_gb10_quarter_brick_10_M_mamba3_b_inv_rms[local_gb10_quarter_brick_10_M_rank, (local_gb10_quarter_brick_10_M_group_state_flat // 64)] * path_c_float32_abi_bank[108195952 + ((((local_gb10_quarter_brick_10_M_rank * 8 + (local_gb10_quarter_brick_10_M_group_state_flat // 64)) * 64 + (local_gb10_quarter_brick_10_M_group_state_flat % 64))) % 2048)]) + path_c_float32_abi_bank[108198000 + ((((local_gb10_quarter_brick_10_M_rank * 8 + (local_gb10_quarter_brick_10_M_group_state_flat // 64)) * 64 + (local_gb10_quarter_brick_10_M_group_state_flat % 64))) % 2048)])
                    local_gb10_quarter_brick_10_M_mamba3_c_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] = local_gb10_quarter_brick_10_M_mamba3_c_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] + ((local_gb10_quarter_brick_10_M_mamba3_conv_vec[9216 + ((local_gb10_quarter_brick_10_M_rank * 8 + (local_gb10_quarter_brick_10_M_group_state_flat // 64)) * 64 + (local_gb10_quarter_brick_10_M_group_state_flat % 64))] * local_gb10_quarter_brick_10_M_mamba3_c_inv_rms[local_gb10_quarter_brick_10_M_rank, (local_gb10_quarter_brick_10_M_group_state_flat // 64)] * path_c_float32_abi_bank[108200048 + ((((local_gb10_quarter_brick_10_M_rank * 8 + (local_gb10_quarter_brick_10_M_group_state_flat // 64)) * 64 + (local_gb10_quarter_brick_10_M_group_state_flat % 64))) % 2048)]) + path_c_float32_abi_bank[108202096 + ((((local_gb10_quarter_brick_10_M_rank * 8 + (local_gb10_quarter_brick_10_M_group_state_flat // 64)) * 64 + (local_gb10_quarter_brick_10_M_group_state_flat % 64))) % 2048)])
                local_gb10_quarter_brick_10_M_mamba3_b_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] = (local_gb10_quarter_brick_10_M_mamba3_b_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] / 4.0) * local_gb10_quarter_brick_10_M_mamba3_trap_group[(local_gb10_quarter_brick_10_M_group_state_flat // 64)]
                local_gb10_quarter_brick_10_M_mamba3_c_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] = local_gb10_quarter_brick_10_M_mamba3_c_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] / 4.0
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_group_state_flat in T.serial(lane, 512, step=256):
                if (local_gb10_quarter_brick_10_M_group_state_flat % 64) < 32:
                    if ((local_gb10_quarter_brick_10_M_group_state_flat % 64) % 2) == 0:
                        local_gb10_quarter_brick_10_M_mamba3_b_group[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] = (local_gb10_quarter_brick_10_M_mamba3_b_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] * T.cos(local_gb10_quarter_brick_10_M_mamba3_angle_cumsum[(local_gb10_quarter_brick_10_M_group_state_flat // 64), ((local_gb10_quarter_brick_10_M_group_state_flat % 64) // 2)])) - (local_gb10_quarter_brick_10_M_mamba3_b_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64) + 1] * T.sin(local_gb10_quarter_brick_10_M_mamba3_angle_cumsum[(local_gb10_quarter_brick_10_M_group_state_flat // 64), ((local_gb10_quarter_brick_10_M_group_state_flat % 64) // 2)]))
                        local_gb10_quarter_brick_10_M_mamba3_c_group[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] = (local_gb10_quarter_brick_10_M_mamba3_c_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] * T.cos(local_gb10_quarter_brick_10_M_mamba3_angle_cumsum[(local_gb10_quarter_brick_10_M_group_state_flat // 64), ((local_gb10_quarter_brick_10_M_group_state_flat % 64) // 2)])) - (local_gb10_quarter_brick_10_M_mamba3_c_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64) + 1] * T.sin(local_gb10_quarter_brick_10_M_mamba3_angle_cumsum[(local_gb10_quarter_brick_10_M_group_state_flat // 64), ((local_gb10_quarter_brick_10_M_group_state_flat % 64) // 2)]))
                    else:
                        local_gb10_quarter_brick_10_M_mamba3_b_group[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] = (local_gb10_quarter_brick_10_M_mamba3_b_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64) - 1] * T.sin(local_gb10_quarter_brick_10_M_mamba3_angle_cumsum[(local_gb10_quarter_brick_10_M_group_state_flat // 64), ((local_gb10_quarter_brick_10_M_group_state_flat % 64) // 2)])) + (local_gb10_quarter_brick_10_M_mamba3_b_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] * T.cos(local_gb10_quarter_brick_10_M_mamba3_angle_cumsum[(local_gb10_quarter_brick_10_M_group_state_flat // 64), ((local_gb10_quarter_brick_10_M_group_state_flat % 64) // 2)]))
                        local_gb10_quarter_brick_10_M_mamba3_c_group[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] = (local_gb10_quarter_brick_10_M_mamba3_c_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64) - 1] * T.sin(local_gb10_quarter_brick_10_M_mamba3_angle_cumsum[(local_gb10_quarter_brick_10_M_group_state_flat // 64), ((local_gb10_quarter_brick_10_M_group_state_flat % 64) // 2)])) + (local_gb10_quarter_brick_10_M_mamba3_c_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] * T.cos(local_gb10_quarter_brick_10_M_mamba3_angle_cumsum[(local_gb10_quarter_brick_10_M_group_state_flat // 64), ((local_gb10_quarter_brick_10_M_group_state_flat % 64) // 2)]))
                else:
                    local_gb10_quarter_brick_10_M_mamba3_b_group[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] = local_gb10_quarter_brick_10_M_mamba3_b_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)]
                    local_gb10_quarter_brick_10_M_mamba3_c_group[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)] = local_gb10_quarter_brick_10_M_mamba3_c_raw[(local_gb10_quarter_brick_10_M_group_state_flat // 64), (local_gb10_quarter_brick_10_M_group_state_flat % 64)]
            T.sync_threads()
            # mamba3_scan_policy: external_state_recurrence
            for local_gb10_quarter_brick_10_M_feature in T.serial(lane, 7168, step=256):
                local_gb10_quarter_brick_10_M_mamba3_out_inner[local_gb10_quarter_brick_10_M_feature] = 0.0
                for local_gb10_quarter_brick_10_M_state_idx in T.serial(0, 64):
                    local_gb10_quarter_brick_10_M_mamba3_state_value[0] = (T.exp(local_gb10_quarter_brick_10_M_mamba3_a_vec[(local_gb10_quarter_brick_10_M_feature // 64)] * local_gb10_quarter_brick_10_M_mamba3_dt_vec[(local_gb10_quarter_brick_10_M_feature // 64)]) * path_c_float32_abi_bank[108663008 + (((local_gb10_quarter_brick_10_M_feature // 64) * 4096 + (local_gb10_quarter_brick_10_M_feature % 64) * 64 + local_gb10_quarter_brick_10_M_state_idx) % 458752)]) + (local_gb10_quarter_brick_10_M_mamba3_conv_vec[local_gb10_quarter_brick_10_M_feature] * local_gb10_quarter_brick_10_M_mamba3_b_group[((local_gb10_quarter_brick_10_M_feature // 64) // 14), local_gb10_quarter_brick_10_M_state_idx])
                    path_c_float32_abi_bank[108663008 + (((local_gb10_quarter_brick_10_M_feature // 64) * 4096 + (local_gb10_quarter_brick_10_M_feature % 64) * 64 + local_gb10_quarter_brick_10_M_state_idx) % 458752)] = local_gb10_quarter_brick_10_M_mamba3_state_value[0]
                    local_gb10_quarter_brick_10_M_mamba3_out_inner[local_gb10_quarter_brick_10_M_feature] = local_gb10_quarter_brick_10_M_mamba3_out_inner[local_gb10_quarter_brick_10_M_feature] + (local_gb10_quarter_brick_10_M_mamba3_state_value[0] * local_gb10_quarter_brick_10_M_mamba3_c_group[((local_gb10_quarter_brick_10_M_feature // 64) // 14), local_gb10_quarter_brick_10_M_state_idx])
                local_gb10_quarter_brick_10_M_mamba3_out_inner[local_gb10_quarter_brick_10_M_feature] = (local_gb10_quarter_brick_10_M_mamba3_out_inner[local_gb10_quarter_brick_10_M_feature] + (path_c_float32_abi_bank[108204144 + (((local_gb10_quarter_brick_10_M_feature // 64)) % 112)] * local_gb10_quarter_brick_10_M_mamba3_conv_vec[local_gb10_quarter_brick_10_M_feature])) * local_gb10_quarter_brick_10_M_mamba3_projected_vec[0 + local_gb10_quarter_brick_10_M_feature] * (1.0 / (1.0 + T.exp(-local_gb10_quarter_brick_10_M_mamba3_projected_vec[0 + local_gb10_quarter_brick_10_M_feature])))
            T.sync_threads()
            # mamba3_output_policy: dense_out_projection
            for local_gb10_quarter_brick_10_M_out_dim in T.serial(lane, 3584, step=256):
                local_gb10_quarter_brick_10_M_mamba3_accum[0] = 0.0
                for local_gb10_quarter_brick_10_M_feature in T.serial(0, 7168):
                    local_gb10_quarter_brick_10_M_mamba3_accum[0] = local_gb10_quarter_brick_10_M_mamba3_accum[0] + (local_gb10_quarter_brick_10_M_mamba3_out_inner[local_gb10_quarter_brick_10_M_feature] * path_c_float32_abi_bank[82460672 + ((local_gb10_quarter_brick_10_M_out_dim * 7168 + local_gb10_quarter_brick_10_M_feature))])
                local_gb10_quarter_brick_10_M_delta[(row * 3584 + local_gb10_quarter_brick_10_M_out_dim) % 3584] = local_gb10_quarter_brick_10_M_mamba3_accum[0]
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_history_flat in T.serial(lane, 11264, step=256):
                local_gb10_quarter_brick_10_M_mamba3_conv_history[(local_gb10_quarter_brick_10_M_history_flat // 11264), (local_gb10_quarter_brick_10_M_history_flat % 11264)] = local_gb10_quarter_brick_10_M_mamba3_conv_history[(local_gb10_quarter_brick_10_M_history_flat // 11264) + 1, (local_gb10_quarter_brick_10_M_history_flat % 11264)]
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_conv_ch in T.serial(lane, 11264, step=256):
                local_gb10_quarter_brick_10_M_mamba3_conv_history[1, local_gb10_quarter_brick_10_M_conv_ch] = local_gb10_quarter_brick_10_M_mamba3_projected_vec[7168 + local_gb10_quarter_brick_10_M_conv_ch]
            T.sync_threads()
            # local_gb10_quarter_brick_11_R_residual_norm: residual_rmsnorm
            # local_gb10_quarter_brick_11_R_residual_norm production_fragment_status: production_region_inlined
            # local_gb10_quarter_brick_11_R_residual_norm production_fragment_reason: row-phased descriptor codegen emits the residual bridge, full-row sum-of-squares reduction, inverse RMS, and weighted normalized output without full activation staging
            local_gb10_quarter_brick_11_R_residual_norm_row_sum_sq_partial[lane] = 0.0
            for i in T.serial(row * 3584 + lane, (row + 1) * 3584, step=256):
                local_gb10_quarter_brick_11_R_residual_norm_row_sum_sq_partial[lane] = local_gb10_quarter_brick_11_R_residual_norm_row_sum_sq_partial[lane] + ((path_c_float32_abi_bank[109121760 + (i)] + local_gb10_quarter_brick_10_M_delta[i % 3584]) * (path_c_float32_abi_bank[109121760 + (i)] + local_gb10_quarter_brick_10_M_delta[i % 3584]))
            T.sync_threads()
            if lane == 0:
                local_gb10_quarter_brick_11_R_residual_norm_row_sum_sq[0] = 0.0
                for partial_lane in T.serial(0, 256):
                    local_gb10_quarter_brick_11_R_residual_norm_row_sum_sq[0] = local_gb10_quarter_brick_11_R_residual_norm_row_sum_sq[0] + local_gb10_quarter_brick_11_R_residual_norm_row_sum_sq_partial[partial_lane]
                local_gb10_quarter_brick_11_R_residual_norm_row_inv_rms[0] = T.rsqrt((local_gb10_quarter_brick_11_R_residual_norm_row_sum_sq[0] / 3584.0) + 0.00001)
            T.sync_threads()
            for i in T.serial(row * 3584 + lane, (row + 1) * 3584, step=256):
                local_gb10_quarter_brick_10_M_hidden_after[i % 3584] = (path_c_float32_abi_bank[109121760 + (i)] + local_gb10_quarter_brick_10_M_delta[i % 3584])
                local_gb10_quarter_brick_11_R_residual_norm_hidden[i % 3584] = (path_c_float32_abi_bank[109121760 + (i)] + local_gb10_quarter_brick_10_M_delta[i % 3584]) * local_gb10_quarter_brick_11_R_residual_norm_row_inv_rms[0] * path_c_float32_abi_bank[123801824 + (i % 3584)]
            T.sync_threads()
            # local_gb10_quarter_brick_11_R: m2rnn
            # local_gb10_quarter_brick_11_R production_fragment_status: production_region_inlined
            # local_gb10_quarter_brick_11_R production_fragment_reason: row-phased descriptor codegen fuses M2RNN dense input projection, causal depthwise convolution, mapped state recurrence, gate/RMSNorm, and output projection from the block-level ABI without full activation staging
            # m2rnn_projection_policy: lane_strided_dense_row_local
            for local_gb10_quarter_brick_11_R_proj_dim in T.serial(lane, 226, step=256):
                local_gb10_quarter_brick_11_R_m2rnn_projected_vec[local_gb10_quarter_brick_11_R_proj_dim] = 0.0
                for local_gb10_quarter_brick_11_R_hidden_dim in T.serial(0, 3584):
                    local_gb10_quarter_brick_11_R_m2rnn_projected_vec[local_gb10_quarter_brick_11_R_proj_dim] = local_gb10_quarter_brick_11_R_m2rnn_projected_vec[local_gb10_quarter_brick_11_R_proj_dim] + (local_gb10_quarter_brick_11_R_residual_norm_hidden[(row * 3584 + local_gb10_quarter_brick_11_R_hidden_dim) % 3584] * path_c_float32_abi_bank[123805408 + ((local_gb10_quarter_brick_11_R_proj_dim * 3584 + local_gb10_quarter_brick_11_R_hidden_dim) % 809984)])
            T.sync_threads()
            # m2rnn_conv_policy: lane_strided_causal_depthwise_ring_history
            for local_gb10_quarter_brick_11_R_conv_ch in T.serial(lane, 160, step=256):
                local_gb10_quarter_brick_11_R_m2rnn_conv_vec[local_gb10_quarter_brick_11_R_conv_ch] = path_c_float32_abi_bank[124616032 + ((local_gb10_quarter_brick_11_R_conv_ch) % 160)]
                for local_gb10_quarter_brick_11_R_kernel_pos in T.serial(0, 3):
                    local_gb10_quarter_brick_11_R_m2rnn_conv_vec[local_gb10_quarter_brick_11_R_conv_ch] = local_gb10_quarter_brick_11_R_m2rnn_conv_vec[local_gb10_quarter_brick_11_R_conv_ch] + (local_gb10_quarter_brick_11_R_m2rnn_conv_history[local_gb10_quarter_brick_11_R_kernel_pos, local_gb10_quarter_brick_11_R_conv_ch] * path_c_float32_abi_bank[124615392 + ((local_gb10_quarter_brick_11_R_conv_ch * 4 + local_gb10_quarter_brick_11_R_kernel_pos) % 640)])
                local_gb10_quarter_brick_11_R_m2rnn_conv_vec[local_gb10_quarter_brick_11_R_conv_ch] = local_gb10_quarter_brick_11_R_m2rnn_conv_vec[local_gb10_quarter_brick_11_R_conv_ch] + (local_gb10_quarter_brick_11_R_m2rnn_projected_vec[local_gb10_quarter_brick_11_R_conv_ch] * path_c_float32_abi_bank[124615392 + ((local_gb10_quarter_brick_11_R_conv_ch * 4 + 3) % 640)])
                local_gb10_quarter_brick_11_R_m2rnn_conv_vec[local_gb10_quarter_brick_11_R_conv_ch] = local_gb10_quarter_brick_11_R_m2rnn_conv_vec[local_gb10_quarter_brick_11_R_conv_ch] * (1.0 / (1.0 + T.exp(-local_gb10_quarter_brick_11_R_m2rnn_conv_vec[local_gb10_quarter_brick_11_R_conv_ch])))
            T.sync_threads()
            # m2rnn_recurrence_policy: lane_strided_mapped_state_update
            for local_gb10_quarter_brick_11_R_head in T.serial(lane, 4, step=256):
                local_gb10_quarter_brick_11_R_m2rnn_decay[0] = T.exp(-T.exp(path_c_float32_abi_bank[124616448 + ((local_gb10_quarter_brick_11_R_head) % 4)]) * T.log(1.0 + T.exp(local_gb10_quarter_brick_11_R_m2rnn_projected_vec[160 + (local_gb10_quarter_brick_11_R_head // 2)] + path_c_float32_abi_bank[124616452 + ((local_gb10_quarter_brick_11_R_head) % 4)])))
                for local_gb10_quarter_brick_11_R_kk in T.serial(0, 64):
                    for local_gb10_quarter_brick_11_R_vv in T.serial(0, 16):
                        local_gb10_quarter_brick_11_R_m2rnn_accum[0] = 0.0
                        for local_gb10_quarter_brick_11_R_vv_inner in T.serial(0, 16):
                            local_gb10_quarter_brick_11_R_m2rnn_accum[0] = local_gb10_quarter_brick_11_R_m2rnn_accum[0] + (local_gb10_quarter_brick_11_R_m2rnn_h_state[local_gb10_quarter_brick_11_R_head, local_gb10_quarter_brick_11_R_kk, local_gb10_quarter_brick_11_R_vv_inner] * path_c_float32_abi_bank[124616192 + (((local_gb10_quarter_brick_11_R_head // 4) * 256 + local_gb10_quarter_brick_11_R_vv_inner * 16 + local_gb10_quarter_brick_11_R_vv) % 256)])
                        local_gb10_quarter_brick_11_R_m2rnn_h_next[local_gb10_quarter_brick_11_R_head, local_gb10_quarter_brick_11_R_kk, local_gb10_quarter_brick_11_R_vv] = (local_gb10_quarter_brick_11_R_m2rnn_decay[0] * local_gb10_quarter_brick_11_R_m2rnn_h_state[local_gb10_quarter_brick_11_R_head, local_gb10_quarter_brick_11_R_kk, local_gb10_quarter_brick_11_R_vv]) + ((1.0 - local_gb10_quarter_brick_11_R_m2rnn_decay[0]) * T.tanh(local_gb10_quarter_brick_11_R_m2rnn_accum[0] + (local_gb10_quarter_brick_11_R_m2rnn_conv_vec[64 + ((local_gb10_quarter_brick_11_R_head // 4) * 64) + local_gb10_quarter_brick_11_R_kk] * local_gb10_quarter_brick_11_R_m2rnn_conv_vec[128 + ((local_gb10_quarter_brick_11_R_head // 2) * 16) + local_gb10_quarter_brick_11_R_vv])))
            T.sync_threads()
            # m2rnn_post_policy: lane_strided_residual_gate_norm_out_proj
            for local_gb10_quarter_brick_11_R_feature in T.serial(lane, 64, step=256):
                local_gb10_quarter_brick_11_R_head = local_gb10_quarter_brick_11_R_feature // 16
                local_gb10_quarter_brick_11_R_vv = local_gb10_quarter_brick_11_R_feature % 16
                local_gb10_quarter_brick_11_R_m2rnn_post_vec[local_gb10_quarter_brick_11_R_feature] = 0.0
                for local_gb10_quarter_brick_11_R_kk in T.serial(0, 64):
                    local_gb10_quarter_brick_11_R_m2rnn_post_vec[local_gb10_quarter_brick_11_R_feature] = local_gb10_quarter_brick_11_R_m2rnn_post_vec[local_gb10_quarter_brick_11_R_feature] + (local_gb10_quarter_brick_11_R_m2rnn_conv_vec[0 + ((local_gb10_quarter_brick_11_R_head // 4) * 64) + local_gb10_quarter_brick_11_R_kk] * local_gb10_quarter_brick_11_R_m2rnn_h_next[local_gb10_quarter_brick_11_R_head, local_gb10_quarter_brick_11_R_kk, local_gb10_quarter_brick_11_R_vv])
                local_gb10_quarter_brick_11_R_m2rnn_post_vec[local_gb10_quarter_brick_11_R_feature] = (local_gb10_quarter_brick_11_R_m2rnn_post_vec[local_gb10_quarter_brick_11_R_feature] + (local_gb10_quarter_brick_11_R_m2rnn_conv_vec[128 + ((local_gb10_quarter_brick_11_R_head // 2) * 16) + local_gb10_quarter_brick_11_R_vv] * path_c_float32_abi_bank[124616456 + ((local_gb10_quarter_brick_11_R_head * 16 + local_gb10_quarter_brick_11_R_vv) % 64)])) * local_gb10_quarter_brick_11_R_m2rnn_projected_vec[162 + (local_gb10_quarter_brick_11_R_feature // 1)] * (1.0 / (1.0 + T.exp(-local_gb10_quarter_brick_11_R_m2rnn_projected_vec[162 + (local_gb10_quarter_brick_11_R_feature // 1)])))
            T.sync_threads()
            for local_gb10_quarter_brick_11_R_state_idx in T.serial(lane, 4096, step=256):
                local_gb10_quarter_brick_11_R_head = local_gb10_quarter_brick_11_R_state_idx // 1024
                local_gb10_quarter_brick_11_R_kk = (local_gb10_quarter_brick_11_R_state_idx // 16) % 64
                local_gb10_quarter_brick_11_R_vv = local_gb10_quarter_brick_11_R_state_idx % 16
                local_gb10_quarter_brick_11_R_m2rnn_h_state[local_gb10_quarter_brick_11_R_head, local_gb10_quarter_brick_11_R_kk, local_gb10_quarter_brick_11_R_vv] = local_gb10_quarter_brick_11_R_m2rnn_h_next[local_gb10_quarter_brick_11_R_head, local_gb10_quarter_brick_11_R_kk, local_gb10_quarter_brick_11_R_vv]
            local_gb10_quarter_brick_11_R_m2rnn_sum_sq_partial[lane] = 0.0
            for local_gb10_quarter_brick_11_R_feature in T.serial(lane, 64, step=256):
                local_gb10_quarter_brick_11_R_m2rnn_sum_sq_partial[lane] = local_gb10_quarter_brick_11_R_m2rnn_sum_sq_partial[lane] + (local_gb10_quarter_brick_11_R_m2rnn_post_vec[local_gb10_quarter_brick_11_R_feature] * local_gb10_quarter_brick_11_R_m2rnn_post_vec[local_gb10_quarter_brick_11_R_feature])
            T.sync_threads()
            if lane == 0:
                local_gb10_quarter_brick_11_R_m2rnn_sum_sq[0] = 0.0
                for local_gb10_quarter_brick_11_R_partial_lane in T.serial(0, 256):
                    local_gb10_quarter_brick_11_R_m2rnn_sum_sq[0] = local_gb10_quarter_brick_11_R_m2rnn_sum_sq[0] + local_gb10_quarter_brick_11_R_m2rnn_sum_sq_partial[local_gb10_quarter_brick_11_R_partial_lane]
                local_gb10_quarter_brick_11_R_m2rnn_inv_rms[0] = T.rsqrt((local_gb10_quarter_brick_11_R_m2rnn_sum_sq[0] / 64.0) + 0.00001)
            T.sync_threads()
            for local_gb10_quarter_brick_11_R_out_dim in T.serial(lane, 3584, step=256):
                local_gb10_quarter_brick_11_R_m2rnn_accum[0] = 0.0
                for local_gb10_quarter_brick_11_R_feature in T.serial(0, 64):
                    local_gb10_quarter_brick_11_R_m2rnn_accum[0] = local_gb10_quarter_brick_11_R_m2rnn_accum[0] + (local_gb10_quarter_brick_11_R_m2rnn_post_vec[local_gb10_quarter_brick_11_R_feature] * local_gb10_quarter_brick_11_R_m2rnn_inv_rms[0] * path_c_float32_abi_bank[124616520 + ((local_gb10_quarter_brick_11_R_feature) % 64)] * path_c_float32_abi_bank[124616584 + ((local_gb10_quarter_brick_11_R_out_dim * 64 + local_gb10_quarter_brick_11_R_feature) % 229376)])
                local_gb10_quarter_brick_11_R_delta[(row * 3584 + local_gb10_quarter_brick_11_R_out_dim) % 3584] = local_gb10_quarter_brick_11_R_m2rnn_accum[0]
            T.sync_threads()
            for local_gb10_quarter_brick_11_R_state_idx in T.serial(lane, 320, step=256):
                local_gb10_quarter_brick_11_R_hist = local_gb10_quarter_brick_11_R_state_idx // 160
                local_gb10_quarter_brick_11_R_conv_ch = local_gb10_quarter_brick_11_R_state_idx % 160
                local_gb10_quarter_brick_11_R_m2rnn_conv_history[local_gb10_quarter_brick_11_R_hist, local_gb10_quarter_brick_11_R_conv_ch] = local_gb10_quarter_brick_11_R_m2rnn_conv_history[local_gb10_quarter_brick_11_R_hist + 1, local_gb10_quarter_brick_11_R_conv_ch]
            for local_gb10_quarter_brick_11_R_conv_ch in T.serial(lane, 160, step=256):
                local_gb10_quarter_brick_11_R_m2rnn_conv_history[2, local_gb10_quarter_brick_11_R_conv_ch] = local_gb10_quarter_brick_11_R_m2rnn_projected_vec[local_gb10_quarter_brick_11_R_conv_ch]
            T.sync_threads()
            # local_gb10_quarter_brick_12_A_residual_norm: residual_rmsnorm
            # local_gb10_quarter_brick_12_A_residual_norm production_fragment_status: production_region_inlined
            # local_gb10_quarter_brick_12_A_residual_norm production_fragment_reason: row-phased descriptor codegen emits the residual bridge, full-row sum-of-squares reduction, inverse RMS, and weighted normalized output without full activation staging
            local_gb10_quarter_brick_12_A_residual_norm_row_sum_sq_partial[lane] = 0.0
            for i in T.serial(row * 3584 + lane, (row + 1) * 3584, step=256):
                local_gb10_quarter_brick_12_A_residual_norm_row_sum_sq_partial[lane] = local_gb10_quarter_brick_12_A_residual_norm_row_sum_sq_partial[lane] + ((local_gb10_quarter_brick_10_M_hidden_after[i % 3584] + local_gb10_quarter_brick_11_R_delta[i % 3584]) * (local_gb10_quarter_brick_10_M_hidden_after[i % 3584] + local_gb10_quarter_brick_11_R_delta[i % 3584]))
            T.sync_threads()
            if lane == 0:
                local_gb10_quarter_brick_12_A_residual_norm_row_sum_sq[0] = 0.0
                for partial_lane in T.serial(0, 256):
                    local_gb10_quarter_brick_12_A_residual_norm_row_sum_sq[0] = local_gb10_quarter_brick_12_A_residual_norm_row_sum_sq[0] + local_gb10_quarter_brick_12_A_residual_norm_row_sum_sq_partial[partial_lane]
                local_gb10_quarter_brick_12_A_residual_norm_row_inv_rms[0] = T.rsqrt((local_gb10_quarter_brick_12_A_residual_norm_row_sum_sq[0] / 3584.0) + 0.00001)
            T.sync_threads()
            for i in T.serial(row * 3584 + lane, (row + 1) * 3584, step=256):
                path_c_float32_abi_bank[124854120 + (i)] = (local_gb10_quarter_brick_10_M_hidden_after[i % 3584] + local_gb10_quarter_brick_11_R_delta[i % 3584])
                local_gb10_quarter_brick_12_A_residual_norm_hidden[i % 3584] = (local_gb10_quarter_brick_10_M_hidden_after[i % 3584] + local_gb10_quarter_brick_11_R_delta[i % 3584]) * local_gb10_quarter_brick_12_A_residual_norm_row_inv_rms[0] * path_c_float32_abi_bank[124850536 + (i % 3584)]
            T.sync_threads()
            # local_gb10_quarter_brick_12_A_qkv_projection: attention_qkv_projection
            # local_gb10_quarter_brick_12_A_qkv_projection production_fragment_status: production_region_inlined
            # local_gb10_quarter_brick_12_A_qkv_projection production_fragment_reason: row-phased descriptor codegen emits real q/sparse-kv dot-products, split-half RoPE, per-head FP8 scaling, uint8 FP8 storage, and full-window causal sparse indices without full activation staging
            local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_vec = T.alloc_local((128,), "float32")
            local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_vec = T.alloc_local((128,), "float32")
            # fp8_prepare_policy: lane_strided_row_head_reduction
            for local_gb10_quarter_brick_12_A_qkv_projection_q_head in T.serial(lane, 28, step=256):
                local_gb10_quarter_brick_12_A_qkv_projection_q_scale[local_gb10_quarter_brick_12_A_qkv_projection_q_head] = 0.0
                for local_gb10_quarter_brick_12_A_qkv_projection_d in T.serial(0, 128):
                    local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d] = 0.0
                for local_gb10_quarter_brick_12_A_qkv_projection_h in T.serial(0, 3584):
                    local_gb10_quarter_brick_12_A_qkv_projection_src_i = row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_h
                    for local_gb10_quarter_brick_12_A_qkv_projection_d in T.serial(0, 128):
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d] = local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d] + (local_gb10_quarter_brick_12_A_residual_norm_hidden[(local_gb10_quarter_brick_12_A_qkv_projection_src_i) % 3584] * path_c_float32_abi_bank[139534184 + (((local_gb10_quarter_brick_12_A_qkv_projection_q_head * 128 + local_gb10_quarter_brick_12_A_qkv_projection_d) * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_h) % 12845056)])
                for local_gb10_quarter_brick_12_A_qkv_projection_d in T.serial(0, 128):
                    local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected[0] = local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d]
                    if local_gb10_quarter_brick_12_A_qkv_projection_d < 64:
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_pair[0] = local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d + 64]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0] = T.cast(row, "float32") * path_c_float32_abi_bank[165224296 + ((local_gb10_quarter_brick_12_A_qkv_projection_d) % 64)]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_q_prepared[0] = (local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected[0] * T.cos(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0])) + (local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_pair[0] * T.sin(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0]))
                    else:
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_pair[0] = local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d - 64]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0] = T.cast(row, "float32") * path_c_float32_abi_bank[165224296 + ((local_gb10_quarter_brick_12_A_qkv_projection_d - 64) % 64)]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_q_prepared[0] = (local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected[0] * T.cos(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0])) - (local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_pair[0] * T.sin(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0]))
                    local_gb10_quarter_brick_12_A_qkv_projection_q_scale[local_gb10_quarter_brick_12_A_qkv_projection_q_head] = T.max(local_gb10_quarter_brick_12_A_qkv_projection_q_scale[local_gb10_quarter_brick_12_A_qkv_projection_q_head], T.abs(T.cast(local_gb10_quarter_brick_12_A_qkv_projection_attention_q_prepared[0], "float32")))
                local_gb10_quarter_brick_12_A_qkv_projection_q_scale[local_gb10_quarter_brick_12_A_qkv_projection_q_head] = T.max(local_gb10_quarter_brick_12_A_qkv_projection_q_scale[local_gb10_quarter_brick_12_A_qkv_projection_q_head] * T.cast(0.002232142857142857, "float32"), T.cast(1.0e-12, "float32"))
                for local_gb10_quarter_brick_12_A_qkv_projection_d in T.serial(0, 128):
                    local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected[0] = local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d]
                    if local_gb10_quarter_brick_12_A_qkv_projection_d < 64:
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_pair[0] = local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d + 64]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0] = T.cast(row, "float32") * path_c_float32_abi_bank[165224296 + ((local_gb10_quarter_brick_12_A_qkv_projection_d) % 64)]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_q_prepared[0] = (local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected[0] * T.cos(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0])) + (local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_pair[0] * T.sin(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0]))
                    else:
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_pair[0] = local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d - 64]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0] = T.cast(row, "float32") * path_c_float32_abi_bank[165224296 + ((local_gb10_quarter_brick_12_A_qkv_projection_d - 64) % 64)]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_q_prepared[0] = (local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected[0] * T.cos(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0])) - (local_gb10_quarter_brick_12_A_qkv_projection_attention_q_projected_pair[0] * T.sin(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0]))
                    local_gb10_quarter_brick_12_A_qkv_projection_q_fp8[local_gb10_quarter_brick_12_A_qkv_projection_q_head * 128 + local_gb10_quarter_brick_12_A_qkv_projection_d] = float_to_fp8_e4m3fn_bits(T.min(T.max((T.cast(local_gb10_quarter_brick_12_A_qkv_projection_attention_q_prepared[0], "float32") / local_gb10_quarter_brick_12_A_qkv_projection_q_scale[local_gb10_quarter_brick_12_A_qkv_projection_q_head]), T.cast(-448.0, "float32")), T.cast(448.0, "float32")))
            for local_gb10_quarter_brick_12_A_qkv_projection_kv_head in T.serial(lane, 28, step=256):
                path_c_float32_abi_bank[165224360 + (((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head) // 3584) * 28 + ((((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head) % 3584) // 128) % 28))] = 0.0
                for local_gb10_quarter_brick_12_A_qkv_projection_d in T.serial(0, 128):
                    local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d] = 0.0
                for local_gb10_quarter_brick_12_A_qkv_projection_h in T.serial(0, 3584):
                    local_gb10_quarter_brick_12_A_qkv_projection_src_i = row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_h
                    for local_gb10_quarter_brick_12_A_qkv_projection_d in T.serial(0, 128):
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d] = local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d] + (local_gb10_quarter_brick_12_A_residual_norm_hidden[(local_gb10_quarter_brick_12_A_qkv_projection_src_i) % 3584] * path_c_float32_abi_bank[152379240 + (((local_gb10_quarter_brick_12_A_qkv_projection_kv_head * 128 + local_gb10_quarter_brick_12_A_qkv_projection_d) * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_h) % 12845056)])
                for local_gb10_quarter_brick_12_A_qkv_projection_d in T.serial(0, 128):
                    local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected[0] = local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d]
                    if local_gb10_quarter_brick_12_A_qkv_projection_d < 64:
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_pair[0] = local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d + 64]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0] = T.cast(row, "float32") * path_c_float32_abi_bank[165224296 + ((local_gb10_quarter_brick_12_A_qkv_projection_d) % 64)]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_prepared[0] = (local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected[0] * T.cos(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0])) + (local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_pair[0] * T.sin(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0]))
                    else:
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_pair[0] = local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d - 64]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0] = T.cast(row, "float32") * path_c_float32_abi_bank[165224296 + ((local_gb10_quarter_brick_12_A_qkv_projection_d - 64) % 64)]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_prepared[0] = (local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected[0] * T.cos(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0])) - (local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_pair[0] * T.sin(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0]))
                    path_c_float32_abi_bank[165224360 + (((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head) // 3584) * 28 + ((((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head) % 3584) // 128) % 28))] = T.max(path_c_float32_abi_bank[165224360 + (((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head) // 3584) * 28 + ((((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head) % 3584) // 128) % 28))], T.abs(T.cast(local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_prepared[0], "float32")))
                path_c_float32_abi_bank[165224360 + (((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head) // 3584) * 28 + ((((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head) % 3584) // 128) % 28))] = T.max(path_c_float32_abi_bank[165224360 + (((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head) // 3584) * 28 + ((((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head) % 3584) // 128) % 28))] * T.cast(0.002232142857142857, "float32"), T.cast(1.0e-12, "float32"))
                for local_gb10_quarter_brick_12_A_qkv_projection_d in T.serial(0, 128):
                    local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected[0] = local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d]
                    if local_gb10_quarter_brick_12_A_qkv_projection_d < 64:
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_pair[0] = local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d + 64]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0] = T.cast(row, "float32") * path_c_float32_abi_bank[165224296 + ((local_gb10_quarter_brick_12_A_qkv_projection_d) % 64)]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_prepared[0] = (local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected[0] * T.cos(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0])) + (local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_pair[0] * T.sin(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0]))
                    else:
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_pair[0] = local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_vec[local_gb10_quarter_brick_12_A_qkv_projection_d - 64]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0] = T.cast(row, "float32") * path_c_float32_abi_bank[165224296 + ((local_gb10_quarter_brick_12_A_qkv_projection_d - 64) % 64)]
                        local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_prepared[0] = (local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected[0] * T.cos(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0])) - (local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_projected_pair[0] * T.sin(local_gb10_quarter_brick_12_A_qkv_projection_attention_rope_phase[0]))
                    path_c_uint8_abi_bank[row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head * 128 + local_gb10_quarter_brick_12_A_qkv_projection_d] = float_to_fp8_e4m3fn_bits(T.min(T.max((T.cast(local_gb10_quarter_brick_12_A_qkv_projection_attention_kv_prepared[0], "float32") / path_c_float32_abi_bank[165224360 + (((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head) // 3584) * 28 + ((((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_kv_head) % 3584) // 128) % 28))]), T.cast(-448.0, "float32")), T.cast(448.0, "float32")))
            for local_gb10_quarter_brick_12_A_qkv_projection_indices_flat in T.serial(lane, 448, step=256):
                local_gb10_quarter_brick_12_A_qkv_projection_kv_head = local_gb10_quarter_brick_12_A_qkv_projection_indices_flat // 16
                local_gb10_quarter_brick_12_A_qkv_projection_k_top = local_gb10_quarter_brick_12_A_qkv_projection_indices_flat % 16
                if row >= local_gb10_quarter_brick_12_A_qkv_projection_k_top:
                    local_gb10_quarter_brick_12_A_qkv_projection_indices[local_gb10_quarter_brick_12_A_qkv_projection_kv_head * 16 + local_gb10_quarter_brick_12_A_qkv_projection_k_top] = row - local_gb10_quarter_brick_12_A_qkv_projection_k_top
                else:
                    local_gb10_quarter_brick_12_A_qkv_projection_indices[local_gb10_quarter_brick_12_A_qkv_projection_kv_head * 16 + local_gb10_quarter_brick_12_A_qkv_projection_k_top] = -1
            T.sync_threads()
            # local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply: sparse_mla_fp8_apply
            # local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply production_fragment_status: production_region_inlined
            # local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply production_fragment_reason: row-phased descriptor codegen emits prepared-FP8 sparse attention apply with score max/sumexp, row-local cached top-k indices, weighted KV values, invalid-index score sentinels, attention out-projection, and LSE from the same softmax stats without full activation staging
            # sparse_mla_fp8_apply_policy: lane_strided_context_and_out_projection
            local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_weights = T.alloc_local((16,), "float32")
            local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_indices = T.alloc_local((16,), "int32")
            local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sink_enabled[0] = T.cast(T.cast(path_c_int32_abi_bank[0], "float32") != 0, "float32")
            for local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head_loop in T.serial(lane, 28, step=256):
                local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head_loop
                local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_q_head[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head[0]
                local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_kv_head[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head[0] // 1
                for local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_k_top in T.serial(0, 16):
                    local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_indices[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_k_top] = local_gb10_quarter_brick_12_A_qkv_projection_indices[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_kv_head[0] * 16 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_k_top]
                local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_max[0] = T.float32(-3.4028234663852886e38)
                for local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_k_top in T.serial(0, 16):
                    local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_weights[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_k_top] = 0.0
                    local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_indices[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_k_top]
                    if local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] >= 0 and local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] < 4096:
                        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] = 0.0
                        for local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_dot_dim in T.serial(0, 128):
                            local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] + (fp8_e4m3fn_to_float(local_gb10_quarter_brick_12_A_qkv_projection_q_fp8[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head[0] * 128 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_dot_dim]) * fp8_e4m3fn_to_float(path_c_uint8_abi_bank[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] * 3584 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_kv_head[0] * 128 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_dot_dim]))
                        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] * local_gb10_quarter_brick_12_A_qkv_projection_q_scale[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head[0]] * path_c_float32_abi_bank[165224360 + (((local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] * 28 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_kv_head[0]) // 3584) * 28 + ((((local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] * 28 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_kv_head[0]) % 3584) // 128) % 28))] * path_c_float32_abi_bank[165339048 + (0)]
                    else:
                        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] = T.float32(-3.4028234663852886e38)
                        if local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] > local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_max[0]:
                            local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_max[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0]
                if local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sink_enabled[0] != 0.0:
                    if path_c_float32_abi_bank[165339049 + ((local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_q_head[0]) % 28)] > local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_max[0]:
                        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_max[0] = path_c_float32_abi_bank[165339049 + ((local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_q_head[0]) % 28)]
                local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sumexp[0] = 0.0
                for local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_k_top in T.serial(0, 16):
                    local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_indices[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_k_top]
                    if local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] >= 0 and local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] < 4096:
                        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] = 0.0
                        for local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_dot_dim in T.serial(0, 128):
                            local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] + (fp8_e4m3fn_to_float(local_gb10_quarter_brick_12_A_qkv_projection_q_fp8[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head[0] * 128 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_dot_dim]) * fp8_e4m3fn_to_float(path_c_uint8_abi_bank[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] * 3584 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_kv_head[0] * 128 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_dot_dim]))
                        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] * local_gb10_quarter_brick_12_A_qkv_projection_q_scale[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head[0]] * path_c_float32_abi_bank[165224360 + (((local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] * 28 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_kv_head[0]) // 3584) * 28 + ((((local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] * 28 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_kv_head[0]) % 3584) // 128) % 28))] * path_c_float32_abi_bank[165339048 + (0)]
                    else:
                        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] = T.float32(-3.4028234663852886e38)
                        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_weight[0] = T.exp(local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_accum[0] - local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_max[0])
                        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_weights[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_k_top] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_weight[0]
                        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sumexp[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sumexp[0] + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_weight[0]
                if local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sink_enabled[0] != 0.0:
                    local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sumexp[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sumexp[0] + T.exp(path_c_float32_abi_bank[165339049 + ((local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_q_head[0]) % 28)] - local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_max[0])
                path_c_float32_abi_bank[192864197 + (((row * 28 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head[0]) // 3584) * 28 + (((row * 28 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head[0]) % 3584) // 128))] = 0.0
                if local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sumexp[0] > 0.0:
                    path_c_float32_abi_bank[192864197 + (((row * 28 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head[0]) // 3584) * 28 + (((row * 28 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head[0]) % 3584) // 128))] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_max[0] + T.log(local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sumexp[0])
                for local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_dim_loop in T.serial(0, 128):
                    local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_dim[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_dim_loop
                    local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_value_accum[0] = 0.0
                    for local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_k_top in T.serial(0, 16):
                        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_indices[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_k_top]
                        if local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] >= 0 and local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] < 4096:
                            local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_value_accum[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_value_accum[0] + (local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_score_weights[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_k_top] * fp8_e4m3fn_to_float(path_c_uint8_abi_bank[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] * 3584 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_kv_head[0] * 128 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_dim[0]]) * path_c_float32_abi_bank[165224360 + (((local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] * 28 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_kv_head[0]) // 3584) * 28 + ((((local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_index[0] * 28 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_kv_head[0]) % 3584) // 128) % 28))])
                    local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_context_accum[0] = 0.0
                    if local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sumexp[0] > 0.0:
                        local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_context_accum[0] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_value_accum[0] / local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sumexp[0]
                    local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_context_values[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_head_loop * 128 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_dim_loop] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_context_accum[0]
            T.sync_threads()
            for local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out_dim_loop in T.serial(lane, 3584, step=256):
                path_c_float32_abi_bank[178184133 + ((row * 3584 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out_dim_loop))] = 0.0
                for local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_dim_loop in T.serial(0, 3584):
                    path_c_float32_abi_bank[178184133 + ((row * 3584 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out_dim_loop))] = path_c_float32_abi_bank[178184133 + ((row * 3584 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out_dim_loop))] + (local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_context_values[local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_dim_loop] * path_c_float32_abi_bank[165339077 + (((local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out_dim_loop) * 3584 + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_source_dim_loop) % 12845056)])
            T.sync_threads()
        for h in T.serial(lane, 3584, step=256):
            path_c_float32_abi_bank[260874245 + ((h) % 3584)] = 0.0
        T.sync_threads()
        for h in T.serial(lane, 3584, step=256):
            path_c_float32_abi_bank[276602541 + ((h) % 3584)] = 0.0
        T.sync_threads()
        # backward_policy: row_phased_hidden_recompute
        for row in T.serial(0, 4096):
            for i in T.serial(row * 3584 + lane, (row + 1) * 3584, step=256):
                # local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd: sparse_mla_fp8_apply_bwd
                # local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd production_fragment_status: production_region_inlined
                # local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd production_fragment_reason: row-phased descriptor codegen emits Sparse-MLA apply backward owner outputs for prepared q/kv FP8 values, prepared scales, and attention out-projection gradients without exposing q/kv prepared gradients as external ABI
                local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd_q_value[0] = fp8_e4m3fn_to_float(local_gb10_quarter_brick_12_A_qkv_projection_q_fp8[(i) % 3584]) * local_gb10_quarter_brick_12_A_qkv_projection_q_scale[((i) % 3584) // 128]
                local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd_kv_value[0] = fp8_e4m3fn_to_float(path_c_uint8_abi_bank[i]) * path_c_float32_abi_bank[165224360 + (((i) // 3584) * 28 + ((((i) % 3584) // 128) % 28))]
                local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd_apply_grad[0] = path_c_float32_abi_bank[192978885 + ((i))] * path_c_float32_abi_bank[165339077 + ((i) % 12845056)] * path_c_float32_abi_bank[165339048 + (0)]
                local_gb10_quarter_brick_12_A_qkv_projection_q_fp8_grad[i % 3584] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd_apply_grad[0] * local_gb10_quarter_brick_12_A_qkv_projection_q_scale[((i) % 3584) // 128]
                local_gb10_quarter_brick_12_A_qkv_projection_q_scale_grad[(i % 3584) // 128] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd_apply_grad[0] * fp8_e4m3fn_to_float(local_gb10_quarter_brick_12_A_qkv_projection_q_fp8[(i) % 3584])
                local_gb10_quarter_brick_12_A_qkv_projection_kv_fp8_grad[i % 3584] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd_apply_grad[0] * path_c_float32_abi_bank[165224360 + (((i) // 3584) * 28 + ((((i) % 3584) // 128) % 28))]
                local_gb10_quarter_brick_12_A_qkv_projection_kv_scale_grad[((i % 3584) // 128) % 28] = local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd_apply_grad[0] * fp8_e4m3fn_to_float(path_c_uint8_abi_bank[i])
                path_c_float32_abi_bank[207658949 + (i % 12845056)] = path_c_float32_abi_bank[192978885 + ((i))] * (local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd_q_value[0] + local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd_kv_value[0])
            T.sync_threads()
            # local_gb10_quarter_brick_12_A_qkv_projection_bwd: attention_qkv_projection_bwd
            # local_gb10_quarter_brick_12_A_qkv_projection_bwd production_fragment_status: production_region_inlined
            # local_gb10_quarter_brick_12_A_qkv_projection_bwd production_fragment_reason: row-phased descriptor codegen emits attention Q/KV projection backward weight, bias, hidden, and RoPE owner gradients from block-level ABI and row-local prepared-FP8 gradients without full activation staging
            # attention_qkv_projection_bwd_policy: lane_strided_weight_bias_rope_hidden
            if row == 0:
                for local_gb10_quarter_brick_12_A_qkv_projection_bwd_grad_flat in T.serial(lane, 12845056, step=256):
                    path_c_float32_abi_bank[220504005 + ((local_gb10_quarter_brick_12_A_qkv_projection_bwd_grad_flat) % 12845056)] = 0.0
                for local_gb10_quarter_brick_12_A_qkv_projection_bwd_grad_flat in T.serial(lane, 12845056, step=256):
                    path_c_float32_abi_bank[233349061 + ((local_gb10_quarter_brick_12_A_qkv_projection_bwd_grad_flat) % 12845056)] = 0.0
                for local_gb10_quarter_brick_12_A_qkv_projection_bwd_rope_d in T.serial(lane, 64, step=256):
                    path_c_float32_abi_bank[246194117 + ((local_gb10_quarter_brick_12_A_qkv_projection_bwd_rope_d) % 64)] = 0.0
            T.sync_threads()
            local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_q_grad = T.alloc_local((1,), "float32")
            local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_kv_grad = T.alloc_local((1,), "float32")
            local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_rope_grad = T.alloc_local((1,), "float32")
            for local_gb10_quarter_brick_12_A_qkv_projection_bwd_q_flat in T.serial(lane, 3584, step=256):
                local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_q_grad[0] = local_gb10_quarter_brick_12_A_qkv_projection_q_fp8_grad[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_q_flat) % 3584] + local_gb10_quarter_brick_12_A_qkv_projection_q_scale_grad[((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_q_flat // 128) % 3584) // 128]
                for local_gb10_quarter_brick_12_A_qkv_projection_bwd_h in T.serial(0, 3584):
                    path_c_float32_abi_bank[220504005 + (((local_gb10_quarter_brick_12_A_qkv_projection_bwd_q_flat) * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 12845056)] = path_c_float32_abi_bank[220504005 + (((local_gb10_quarter_brick_12_A_qkv_projection_bwd_q_flat) * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 12845056)] + (local_gb10_quarter_brick_12_A_residual_norm_hidden[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 3584] * local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_q_grad[0])
            T.sync_threads()
            for local_gb10_quarter_brick_12_A_qkv_projection_bwd_kv_flat in T.serial(lane, 3584, step=256):
                local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_kv_grad[0] = local_gb10_quarter_brick_12_A_qkv_projection_kv_fp8_grad[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_kv_flat) % 3584] + local_gb10_quarter_brick_12_A_qkv_projection_kv_scale_grad[(((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_kv_flat // 128) % 3584) // 128) % 28]
                for local_gb10_quarter_brick_12_A_qkv_projection_bwd_h in T.serial(0, 3584):
                    path_c_float32_abi_bank[233349061 + (((local_gb10_quarter_brick_12_A_qkv_projection_bwd_kv_flat) * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 12845056)] = path_c_float32_abi_bank[233349061 + (((local_gb10_quarter_brick_12_A_qkv_projection_bwd_kv_flat) * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 12845056)] + (local_gb10_quarter_brick_12_A_residual_norm_hidden[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 3584] * local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_kv_grad[0])
            T.sync_threads()
            for local_gb10_quarter_brick_12_A_qkv_projection_bwd_h in T.serial(lane, 3584, step=256):
                local_gb10_quarter_brick_12_A_residual_norm_hidden_grad[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 3584] = 0.0
                for local_gb10_quarter_brick_12_A_qkv_projection_bwd_q_flat in T.serial(0, 3584):
                    local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_q_grad[0] = local_gb10_quarter_brick_12_A_qkv_projection_q_fp8_grad[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_q_flat) % 3584] + local_gb10_quarter_brick_12_A_qkv_projection_q_scale_grad[((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_q_flat // 128) % 3584) // 128]
                    local_gb10_quarter_brick_12_A_residual_norm_hidden_grad[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 3584] = local_gb10_quarter_brick_12_A_residual_norm_hidden_grad[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 3584] + (local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_q_grad[0] * path_c_float32_abi_bank[139534184 + (((local_gb10_quarter_brick_12_A_qkv_projection_bwd_q_flat) * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 12845056)])
                for local_gb10_quarter_brick_12_A_qkv_projection_bwd_kv_flat in T.serial(0, 3584):
                    local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_kv_grad[0] = local_gb10_quarter_brick_12_A_qkv_projection_kv_fp8_grad[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_kv_flat) % 3584] + local_gb10_quarter_brick_12_A_qkv_projection_kv_scale_grad[(((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_kv_flat // 128) % 3584) // 128) % 28]
                    local_gb10_quarter_brick_12_A_residual_norm_hidden_grad[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 3584] = local_gb10_quarter_brick_12_A_residual_norm_hidden_grad[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 3584] + (local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_kv_grad[0] * path_c_float32_abi_bank[152379240 + (((local_gb10_quarter_brick_12_A_qkv_projection_bwd_kv_flat) * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_h) % 12845056)])
            T.sync_threads()
            for local_gb10_quarter_brick_12_A_qkv_projection_bwd_rope_d in T.serial(lane, 64, step=256):
                local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_rope_grad[0] = 0.0
                for local_gb10_quarter_brick_12_A_qkv_projection_bwd_q_flat in T.serial(0, 28):
                    local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_q_grad[0] = local_gb10_quarter_brick_12_A_qkv_projection_q_fp8_grad[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_q_flat * 128 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_rope_d) % 3584] + local_gb10_quarter_brick_12_A_qkv_projection_q_scale_grad[((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_q_flat) % 3584) // 128]
                    local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_rope_grad[0] = local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_rope_grad[0] + (local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_q_grad[0] * T.cast(row, "float32"))
                for local_gb10_quarter_brick_12_A_qkv_projection_bwd_kv_flat in T.serial(0, 28):
                    local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_kv_grad[0] = local_gb10_quarter_brick_12_A_qkv_projection_kv_fp8_grad[(row * 3584 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_kv_flat * 128 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_rope_d) % 3584] + local_gb10_quarter_brick_12_A_qkv_projection_kv_scale_grad[(((row * 28 + local_gb10_quarter_brick_12_A_qkv_projection_bwd_kv_flat) % 3584) // 128) % 28]
                    local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_rope_grad[0] = local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_rope_grad[0] + (local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_kv_grad[0] * T.cast(row, "float32"))
                path_c_float32_abi_bank[246194117 + ((local_gb10_quarter_brick_12_A_qkv_projection_bwd_rope_d) % 64)] = path_c_float32_abi_bank[246194117 + ((local_gb10_quarter_brick_12_A_qkv_projection_bwd_rope_d) % 64)] + local_gb10_quarter_brick_12_A_qkv_projection_bwd_attention_rope_grad[0]
            T.sync_threads()
            # local_gb10_quarter_brick_12_A_residual_norm_bwd: residual_rmsnorm_bwd
            # local_gb10_quarter_brick_12_A_residual_norm_bwd production_fragment_status: production_region_inlined
            # local_gb10_quarter_brick_12_A_residual_norm_bwd production_fragment_reason: row-phased descriptor codegen recomputes residual/RMSNorm state from forward inputs and accumulates norm-weight grads without full activation staging
            local_gb10_quarter_brick_12_A_residual_norm_bwd_row_sum_sq_partial[lane] = 0.0
            local_gb10_quarter_brick_12_A_residual_norm_bwd_row_dot_partial[lane] = 0.0
            for i in T.serial(row * 3584 + lane, (row + 1) * 3584, step=256):
                local_gb10_quarter_brick_12_A_residual_norm_bwd_row_sum_sq_partial[lane] = local_gb10_quarter_brick_12_A_residual_norm_bwd_row_sum_sq_partial[lane] + ((local_gb10_quarter_brick_10_M_hidden_after[i % 3584] + local_gb10_quarter_brick_11_R_delta[i % 3584]) * (local_gb10_quarter_brick_10_M_hidden_after[i % 3584] + local_gb10_quarter_brick_11_R_delta[i % 3584]))
                local_gb10_quarter_brick_12_A_residual_norm_bwd_row_dot_partial[lane] = local_gb10_quarter_brick_12_A_residual_norm_bwd_row_dot_partial[lane] + (local_gb10_quarter_brick_12_A_residual_norm_hidden_grad[i % 3584] * path_c_float32_abi_bank[124850536 + (i % 3584)] * (local_gb10_quarter_brick_10_M_hidden_after[i % 3584] + local_gb10_quarter_brick_11_R_delta[i % 3584]))
            T.sync_threads()
            if lane == 0:
                local_gb10_quarter_brick_12_A_residual_norm_bwd_row_sum_sq[0] = 0.0
                local_gb10_quarter_brick_12_A_residual_norm_bwd_row_dot[0] = 0.0
                for partial_lane in T.serial(0, 256):
                    local_gb10_quarter_brick_12_A_residual_norm_bwd_row_sum_sq[0] = local_gb10_quarter_brick_12_A_residual_norm_bwd_row_sum_sq[0] + local_gb10_quarter_brick_12_A_residual_norm_bwd_row_sum_sq_partial[partial_lane]
                    local_gb10_quarter_brick_12_A_residual_norm_bwd_row_dot[0] = local_gb10_quarter_brick_12_A_residual_norm_bwd_row_dot[0] + local_gb10_quarter_brick_12_A_residual_norm_bwd_row_dot_partial[partial_lane]
                local_gb10_quarter_brick_12_A_residual_norm_bwd_row_inv_rms[0] = T.rsqrt((local_gb10_quarter_brick_12_A_residual_norm_bwd_row_sum_sq[0] / 3584.0) + 0.00001)
            T.sync_threads()
            for i in T.serial(row * 3584 + lane, (row + 1) * 3584, step=256):
                local_gb10_quarter_brick_12_A_residual_norm_bwd_row_norm_grad[0] = local_gb10_quarter_brick_12_A_residual_norm_hidden_grad[i % 3584] * path_c_float32_abi_bank[124850536 + (i % 3584)]
                local_gb10_quarter_brick_12_A_residual_norm_bwd_row_total_grad[0] = path_c_float32_abi_bank[246194181 + (i)] + (local_gb10_quarter_brick_12_A_residual_norm_bwd_row_inv_rms[0] * (local_gb10_quarter_brick_12_A_residual_norm_bwd_row_norm_grad[0] - ((local_gb10_quarter_brick_10_M_hidden_after[i % 3584] + local_gb10_quarter_brick_11_R_delta[i % 3584]) * local_gb10_quarter_brick_12_A_residual_norm_bwd_row_dot[0] * local_gb10_quarter_brick_12_A_residual_norm_bwd_row_inv_rms[0] * local_gb10_quarter_brick_12_A_residual_norm_bwd_row_inv_rms[0] / 3584.0)))
                local_gb10_quarter_brick_10_M_hidden_after_grad[i % 3584] = local_gb10_quarter_brick_12_A_residual_norm_bwd_row_total_grad[0]
                local_gb10_quarter_brick_11_R_delta_grad[i % 3584] = local_gb10_quarter_brick_12_A_residual_norm_bwd_row_total_grad[0]
                path_c_float32_abi_bank[260874245 + (i % 3584)] = path_c_float32_abi_bank[260874245 + (i % 3584)] + (local_gb10_quarter_brick_12_A_residual_norm_hidden_grad[i % 3584] * (local_gb10_quarter_brick_10_M_hidden_after[i % 3584] + local_gb10_quarter_brick_11_R_delta[i % 3584]) * local_gb10_quarter_brick_12_A_residual_norm_bwd_row_inv_rms[0])
            T.sync_threads()
            # local_gb10_quarter_brick_11_R_bwd: m2rnn_bwd
            # local_gb10_quarter_brick_11_R_bwd production_fragment_status: production_region_inlined
            # local_gb10_quarter_brick_11_R_bwd production_fragment_reason: row-phased descriptor codegen recomputes M2RNN backward owner outputs from block-level projection, convolution, state, gate, post, h0, and row-local hidden gradients without full activation staging
            # m2rnn_bwd_policy: lane_strided_weight_state_recompute
            local_gb10_quarter_brick_11_R_bwd_m2rnn_stage_grad = T.alloc_local((1,), "float32")
            if row == 0:
                for local_gb10_quarter_brick_11_R_bwd_state_idx in T.serial(lane, 809984, step=256):
                    path_c_float32_abi_bank[260877829 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 809984)] = 0.0
                for local_gb10_quarter_brick_11_R_bwd_state_idx in T.serial(lane, 640, step=256):
                    path_c_float32_abi_bank[261687813 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 640)] = 0.0
                for local_gb10_quarter_brick_11_R_bwd_state_idx in T.serial(lane, 160, step=256):
                    path_c_float32_abi_bank[261688453 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 160)] = 0.0
                for local_gb10_quarter_brick_11_R_bwd_state_idx in T.serial(lane, 256, step=256):
                    path_c_float32_abi_bank[261688613 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 256)] = 0.0
                for local_gb10_quarter_brick_11_R_bwd_state_idx in T.serial(lane, 4, step=256):
                    path_c_float32_abi_bank[261688869 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 4)] = 0.0
                for local_gb10_quarter_brick_11_R_bwd_state_idx in T.serial(lane, 4, step=256):
                    path_c_float32_abi_bank[261688873 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 4)] = 0.0
                for local_gb10_quarter_brick_11_R_bwd_state_idx in T.serial(lane, 64, step=256):
                    path_c_float32_abi_bank[261688877 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 64)] = 0.0
                for local_gb10_quarter_brick_11_R_bwd_state_idx in T.serial(lane, 64, step=256):
                    path_c_float32_abi_bank[261688941 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 64)] = 0.0
                for local_gb10_quarter_brick_11_R_bwd_state_idx in T.serial(lane, 229376, step=256):
                    path_c_float32_abi_bank[261689005 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 229376)] = 0.0
                for local_gb10_quarter_brick_11_R_bwd_state_idx in T.serial(lane, 256, step=256):
                    path_c_float32_abi_bank[261918381 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 4096)] = 0.0
            T.sync_threads()
            for local_gb10_quarter_brick_11_R_bwd_grad_flat in T.serial(lane, 809984, step=256):
                local_gb10_quarter_brick_11_R_bwd_m2rnn_stage_grad[0] = local_gb10_quarter_brick_11_R_delta_grad[(row * 3584 + ((local_gb10_quarter_brick_11_R_bwd_grad_flat // 3584) % 3584)) % 3584] * path_c_float32_abi_bank[123805408 + (((local_gb10_quarter_brick_11_R_bwd_grad_flat // 3584) * 3584 + ((local_gb10_quarter_brick_11_R_bwd_grad_flat // 3584) % 3584)) % 809984)]
                path_c_float32_abi_bank[260877829 + ((local_gb10_quarter_brick_11_R_bwd_grad_flat) % 809984)] = path_c_float32_abi_bank[260877829 + ((local_gb10_quarter_brick_11_R_bwd_grad_flat) % 809984)] + (local_gb10_quarter_brick_11_R_residual_norm_hidden[(row * 3584 + (local_gb10_quarter_brick_11_R_bwd_grad_flat % 3584)) % 3584] * local_gb10_quarter_brick_11_R_bwd_m2rnn_stage_grad[0])
            T.sync_threads()
            for local_gb10_quarter_brick_11_R_bwd_hidden_dim in T.serial(lane, 3584, step=256):
                local_gb10_quarter_brick_11_R_residual_norm_hidden_grad[(row * 3584 + local_gb10_quarter_brick_11_R_bwd_hidden_dim) % 3584] = 0.0
                for local_gb10_quarter_brick_11_R_bwd_proj_dim in T.serial(0, 226):
                    local_gb10_quarter_brick_11_R_bwd_m2rnn_stage_grad[0] = local_gb10_quarter_brick_11_R_delta_grad[(row * 3584 + (local_gb10_quarter_brick_11_R_bwd_proj_dim % 3584)) % 3584] * path_c_float32_abi_bank[123805408 + ((local_gb10_quarter_brick_11_R_bwd_proj_dim * 3584 + (local_gb10_quarter_brick_11_R_bwd_proj_dim % 3584)) % 809984)]
                    local_gb10_quarter_brick_11_R_residual_norm_hidden_grad[(row * 3584 + local_gb10_quarter_brick_11_R_bwd_hidden_dim) % 3584] = local_gb10_quarter_brick_11_R_residual_norm_hidden_grad[(row * 3584 + local_gb10_quarter_brick_11_R_bwd_hidden_dim) % 3584] + (local_gb10_quarter_brick_11_R_bwd_m2rnn_stage_grad[0] * path_c_float32_abi_bank[123805408 + ((local_gb10_quarter_brick_11_R_bwd_proj_dim * 3584 + local_gb10_quarter_brick_11_R_bwd_hidden_dim) % 809984)])
            T.sync_threads()
            for local_gb10_quarter_brick_11_R_bwd_conv_ch in T.serial(lane, 160, step=256):
                path_c_float32_abi_bank[261688453 + ((local_gb10_quarter_brick_11_R_bwd_conv_ch) % 160)] = path_c_float32_abi_bank[261688453 + ((local_gb10_quarter_brick_11_R_bwd_conv_ch) % 160)] + local_gb10_quarter_brick_11_R_delta_grad[(row * 3584 + (local_gb10_quarter_brick_11_R_bwd_conv_ch % 3584)) % 3584]
            T.sync_threads()
            for local_gb10_quarter_brick_11_R_bwd_grad_flat in T.serial(lane, 640, step=256):
                path_c_float32_abi_bank[261687813 + ((local_gb10_quarter_brick_11_R_bwd_grad_flat) % 640)] = path_c_float32_abi_bank[261687813 + ((local_gb10_quarter_brick_11_R_bwd_grad_flat) % 640)] + (local_gb10_quarter_brick_11_R_delta_grad[(row * 3584 + ((local_gb10_quarter_brick_11_R_bwd_grad_flat // 4) % 3584)) % 3584] * local_gb10_quarter_brick_11_R_residual_norm_hidden[(row * 3584 + ((local_gb10_quarter_brick_11_R_bwd_grad_flat // 4) % 3584)) % 3584])
            T.sync_threads()
            for local_gb10_quarter_brick_11_R_bwd_state_idx in T.serial(lane, 256, step=256):
                path_c_float32_abi_bank[261688613 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 256)] = path_c_float32_abi_bank[261688613 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 256)] + (path_c_float32_abi_bank[124845960 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 4096)] * local_gb10_quarter_brick_11_R_delta_grad[(row * 3584 + (local_gb10_quarter_brick_11_R_bwd_state_idx % 3584)) % 3584])
                path_c_float32_abi_bank[261918381 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 4096)] = path_c_float32_abi_bank[261918381 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 4096)] + local_gb10_quarter_brick_11_R_delta_grad[(row * 3584 + (local_gb10_quarter_brick_11_R_bwd_state_idx % 3584)) % 3584]
            T.sync_threads()
            for local_gb10_quarter_brick_11_R_bwd_state_idx in T.serial(lane, 4, step=256):
                path_c_float32_abi_bank[261688869 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 4)] = path_c_float32_abi_bank[261688869 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 4)] + local_gb10_quarter_brick_11_R_delta_grad[(row * 3584 + (local_gb10_quarter_brick_11_R_bwd_state_idx % 3584)) % 3584]
                path_c_float32_abi_bank[261688873 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 4)] = path_c_float32_abi_bank[261688873 + ((local_gb10_quarter_brick_11_R_bwd_state_idx) % 4)] + local_gb10_quarter_brick_11_R_delta_grad[(row * 3584 + (local_gb10_quarter_brick_11_R_bwd_state_idx % 3584)) % 3584]
            T.sync_threads()
            for local_gb10_quarter_brick_11_R_bwd_feature in T.serial(lane, 64, step=256):
                path_c_float32_abi_bank[261688877 + ((local_gb10_quarter_brick_11_R_bwd_feature) % 64)] = path_c_float32_abi_bank[261688877 + ((local_gb10_quarter_brick_11_R_bwd_feature) % 64)] + local_gb10_quarter_brick_11_R_delta_grad[(row * 3584 + (local_gb10_quarter_brick_11_R_bwd_feature % 3584)) % 3584]
                path_c_float32_abi_bank[261688941 + ((local_gb10_quarter_brick_11_R_bwd_feature) % 64)] = path_c_float32_abi_bank[261688941 + ((local_gb10_quarter_brick_11_R_bwd_feature) % 64)] + local_gb10_quarter_brick_11_R_delta_grad[(row * 3584 + (local_gb10_quarter_brick_11_R_bwd_feature % 3584)) % 3584]
            T.sync_threads()
            for local_gb10_quarter_brick_11_R_bwd_grad_flat in T.serial(lane, 229376, step=256):
                path_c_float32_abi_bank[261689005 + ((local_gb10_quarter_brick_11_R_bwd_grad_flat) % 229376)] = path_c_float32_abi_bank[261689005 + ((local_gb10_quarter_brick_11_R_bwd_grad_flat) % 229376)] + (local_gb10_quarter_brick_11_R_residual_norm_hidden[(row * 3584 + (local_gb10_quarter_brick_11_R_bwd_grad_flat // 64)) % 3584] * local_gb10_quarter_brick_11_R_delta_grad[(row * 3584 + (local_gb10_quarter_brick_11_R_bwd_grad_flat // 64)) % 3584])
            T.sync_threads()
            # local_gb10_quarter_brick_11_R_residual_norm_bwd: residual_rmsnorm_bwd
            # local_gb10_quarter_brick_11_R_residual_norm_bwd production_fragment_status: production_region_inlined
            # local_gb10_quarter_brick_11_R_residual_norm_bwd production_fragment_reason: row-phased descriptor codegen recomputes residual/RMSNorm state from forward inputs and accumulates norm-weight grads without full activation staging
            local_gb10_quarter_brick_11_R_residual_norm_bwd_row_sum_sq_partial[lane] = 0.0
            local_gb10_quarter_brick_11_R_residual_norm_bwd_row_dot_partial[lane] = 0.0
            for i in T.serial(row * 3584 + lane, (row + 1) * 3584, step=256):
                local_gb10_quarter_brick_11_R_residual_norm_bwd_row_sum_sq_partial[lane] = local_gb10_quarter_brick_11_R_residual_norm_bwd_row_sum_sq_partial[lane] + ((path_c_float32_abi_bank[109121760 + (i)] + local_gb10_quarter_brick_10_M_delta[i % 3584]) * (path_c_float32_abi_bank[109121760 + (i)] + local_gb10_quarter_brick_10_M_delta[i % 3584]))
                local_gb10_quarter_brick_11_R_residual_norm_bwd_row_dot_partial[lane] = local_gb10_quarter_brick_11_R_residual_norm_bwd_row_dot_partial[lane] + (local_gb10_quarter_brick_11_R_residual_norm_hidden_grad[i % 3584] * path_c_float32_abi_bank[123801824 + (i % 3584)] * (path_c_float32_abi_bank[109121760 + (i)] + local_gb10_quarter_brick_10_M_delta[i % 3584]))
            T.sync_threads()
            if lane == 0:
                local_gb10_quarter_brick_11_R_residual_norm_bwd_row_sum_sq[0] = 0.0
                local_gb10_quarter_brick_11_R_residual_norm_bwd_row_dot[0] = 0.0
                for partial_lane in T.serial(0, 256):
                    local_gb10_quarter_brick_11_R_residual_norm_bwd_row_sum_sq[0] = local_gb10_quarter_brick_11_R_residual_norm_bwd_row_sum_sq[0] + local_gb10_quarter_brick_11_R_residual_norm_bwd_row_sum_sq_partial[partial_lane]
                    local_gb10_quarter_brick_11_R_residual_norm_bwd_row_dot[0] = local_gb10_quarter_brick_11_R_residual_norm_bwd_row_dot[0] + local_gb10_quarter_brick_11_R_residual_norm_bwd_row_dot_partial[partial_lane]
                local_gb10_quarter_brick_11_R_residual_norm_bwd_row_inv_rms[0] = T.rsqrt((local_gb10_quarter_brick_11_R_residual_norm_bwd_row_sum_sq[0] / 3584.0) + 0.00001)
            T.sync_threads()
            for i in T.serial(row * 3584 + lane, (row + 1) * 3584, step=256):
                local_gb10_quarter_brick_11_R_residual_norm_bwd_row_norm_grad[0] = local_gb10_quarter_brick_11_R_residual_norm_hidden_grad[i % 3584] * path_c_float32_abi_bank[123801824 + (i % 3584)]
                local_gb10_quarter_brick_11_R_residual_norm_bwd_row_total_grad[0] = local_gb10_quarter_brick_10_M_hidden_after_grad[i % 3584] + (local_gb10_quarter_brick_11_R_residual_norm_bwd_row_inv_rms[0] * (local_gb10_quarter_brick_11_R_residual_norm_bwd_row_norm_grad[0] - ((path_c_float32_abi_bank[109121760 + (i)] + local_gb10_quarter_brick_10_M_delta[i % 3584]) * local_gb10_quarter_brick_11_R_residual_norm_bwd_row_dot[0] * local_gb10_quarter_brick_11_R_residual_norm_bwd_row_inv_rms[0] * local_gb10_quarter_brick_11_R_residual_norm_bwd_row_inv_rms[0] / 3584.0)))
                path_c_float32_abi_bank[261922477 + (i)] = local_gb10_quarter_brick_11_R_residual_norm_bwd_row_total_grad[0]
                local_gb10_quarter_brick_10_M_delta_grad[i % 3584] = local_gb10_quarter_brick_11_R_residual_norm_bwd_row_total_grad[0]
                path_c_float32_abi_bank[276602541 + (i % 3584)] = path_c_float32_abi_bank[276602541 + (i % 3584)] + (local_gb10_quarter_brick_11_R_residual_norm_hidden_grad[i % 3584] * (path_c_float32_abi_bank[109121760 + (i)] + local_gb10_quarter_brick_10_M_delta[i % 3584]) * local_gb10_quarter_brick_11_R_residual_norm_bwd_row_inv_rms[0])
            T.sync_threads()
            # local_gb10_quarter_brick_10_M_bwd: mamba3_mimo_bwd
            # local_gb10_quarter_brick_10_M_bwd production_fragment_status: production_region_inlined
            # local_gb10_quarter_brick_10_M_bwd production_fragment_reason: row-phased descriptor codegen recomputes Mamba3 backward owner outputs from block-level weights, state, h0, and row-local hidden gradients without full activation staging
            # mamba3_mimo_bwd_policy: lane_strided_weight_state_recompute
            local_gb10_quarter_brick_10_M_bwd_mamba3_stage_grad = T.alloc_local((1,), "float32")
            if row == 0:
                for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 67321856, step=256):
                    path_c_float32_abi_bank[291744941 + ((local_gb10_quarter_brick_10_M_bwd_state_idx))] = 0.0
                for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 25690112, step=256):
                    path_c_float32_abi_bank[359066797 + ((local_gb10_quarter_brick_10_M_bwd_state_idx))] = 0.0
                for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 33792, step=256):
                    path_c_float32_abi_bank[384756909 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 33792)] = 0.0
                for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 11264, step=256):
                    path_c_float32_abi_bank[384790701 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 11264)] = 0.0
                for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 112, step=256):
                    path_c_float32_abi_bank[384801965 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 112)] = 0.0
                for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 2048, step=256):
                    path_c_float32_abi_bank[384802077 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 2048)] = 0.0
                for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 2048, step=256):
                    path_c_float32_abi_bank[384804125 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 2048)] = 0.0
                for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 2048, step=256):
                    path_c_float32_abi_bank[384806173 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 2048)] = 0.0
                for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 2048, step=256):
                    path_c_float32_abi_bank[384808221 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 2048)] = 0.0
                for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 112, step=256):
                    path_c_float32_abi_bank[384810269 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 112)] = 0.0
                for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 458752, step=256):
                    path_c_float32_abi_bank[384810381 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 458752)] = 0.0
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_bwd_grad_flat in T.serial(lane, 67321856, step=256):
                local_gb10_quarter_brick_10_M_bwd_mamba3_stage_grad[0] = local_gb10_quarter_brick_10_M_delta_grad[(row * 3584 + ((local_gb10_quarter_brick_10_M_bwd_grad_flat // 3584) % 3584)) % 3584] * path_c_float32_abi_bank[15138816 + (((local_gb10_quarter_brick_10_M_bwd_grad_flat // 3584) * 3584 + ((local_gb10_quarter_brick_10_M_bwd_grad_flat // 3584) % 3584)))]
                path_c_float32_abi_bank[291744941 + ((local_gb10_quarter_brick_10_M_bwd_grad_flat))] = path_c_float32_abi_bank[291744941 + ((local_gb10_quarter_brick_10_M_bwd_grad_flat))] + (path_c_float32_abi_bank[row * 3584 + (local_gb10_quarter_brick_10_M_bwd_grad_flat % 3584)] * local_gb10_quarter_brick_10_M_bwd_mamba3_stage_grad[0])
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_bwd_hidden_dim in T.serial(lane, 3584, step=256):
                path_c_float32_abi_bank[276606125 + ((row * 3584 + local_gb10_quarter_brick_10_M_bwd_hidden_dim))] = 0.0
                for local_gb10_quarter_brick_10_M_bwd_proj_dim in T.serial(0, 18784):
                    local_gb10_quarter_brick_10_M_bwd_mamba3_stage_grad[0] = local_gb10_quarter_brick_10_M_delta_grad[(row * 3584 + (local_gb10_quarter_brick_10_M_bwd_proj_dim % 3584)) % 3584] * path_c_float32_abi_bank[15138816 + ((local_gb10_quarter_brick_10_M_bwd_proj_dim * 3584 + (local_gb10_quarter_brick_10_M_bwd_proj_dim % 3584)))]
                    path_c_float32_abi_bank[276606125 + ((row * 3584 + local_gb10_quarter_brick_10_M_bwd_hidden_dim))] = path_c_float32_abi_bank[276606125 + ((row * 3584 + local_gb10_quarter_brick_10_M_bwd_hidden_dim))] + (local_gb10_quarter_brick_10_M_bwd_mamba3_stage_grad[0] * path_c_float32_abi_bank[15138816 + ((local_gb10_quarter_brick_10_M_bwd_proj_dim * 3584 + local_gb10_quarter_brick_10_M_bwd_hidden_dim))])
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_bwd_conv_ch in T.serial(lane, 11264, step=256):
                path_c_float32_abi_bank[384790701 + ((local_gb10_quarter_brick_10_M_bwd_conv_ch) % 11264)] = path_c_float32_abi_bank[384790701 + ((local_gb10_quarter_brick_10_M_bwd_conv_ch) % 11264)] + local_gb10_quarter_brick_10_M_delta_grad[(row * 3584 + (local_gb10_quarter_brick_10_M_bwd_conv_ch % 3584)) % 3584]
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_bwd_grad_flat in T.serial(lane, 33792, step=256):
                path_c_float32_abi_bank[384756909 + ((local_gb10_quarter_brick_10_M_bwd_grad_flat) % 33792)] = path_c_float32_abi_bank[384756909 + ((local_gb10_quarter_brick_10_M_bwd_grad_flat) % 33792)] + (path_c_float32_abi_bank[row * 3584 + ((local_gb10_quarter_brick_10_M_bwd_grad_flat // 3) % 3584)] * local_gb10_quarter_brick_10_M_delta_grad[(row * 3584 + ((local_gb10_quarter_brick_10_M_bwd_grad_flat // 3) % 3584)) % 3584])
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 2048, step=256):
                path_c_float32_abi_bank[384802077 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 2048)] = path_c_float32_abi_bank[384802077 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 2048)] + (path_c_float32_abi_bank[14680064 + ((local_gb10_quarter_brick_10_M_bwd_state_idx % 458752) % 458752)] * local_gb10_quarter_brick_10_M_delta_grad[(row * 3584 + (local_gb10_quarter_brick_10_M_bwd_state_idx % 3584)) % 3584])
                path_c_float32_abi_bank[384804125 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 2048)] = path_c_float32_abi_bank[384804125 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 2048)] + local_gb10_quarter_brick_10_M_delta_grad[(row * 3584 + (local_gb10_quarter_brick_10_M_bwd_state_idx % 3584)) % 3584]
                path_c_float32_abi_bank[384806173 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 2048)] = path_c_float32_abi_bank[384806173 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 2048)] + (path_c_float32_abi_bank[14680064 + ((local_gb10_quarter_brick_10_M_bwd_state_idx % 458752) % 458752)] * local_gb10_quarter_brick_10_M_delta_grad[(row * 3584 + (local_gb10_quarter_brick_10_M_bwd_state_idx % 3584)) % 3584])
                path_c_float32_abi_bank[384808221 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 2048)] = path_c_float32_abi_bank[384808221 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 2048)] + local_gb10_quarter_brick_10_M_delta_grad[(row * 3584 + (local_gb10_quarter_brick_10_M_bwd_state_idx % 3584)) % 3584]
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 112, step=256):
                path_c_float32_abi_bank[384801965 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 112)] = path_c_float32_abi_bank[384801965 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 112)] + local_gb10_quarter_brick_10_M_delta_grad[(row * 3584 + (local_gb10_quarter_brick_10_M_bwd_state_idx % 3584)) % 3584]
                path_c_float32_abi_bank[384810269 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 112)] = path_c_float32_abi_bank[384810269 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 112)] + local_gb10_quarter_brick_10_M_delta_grad[(row * 3584 + (local_gb10_quarter_brick_10_M_bwd_state_idx % 3584)) % 3584]
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_bwd_grad_flat in T.serial(lane, 25690112, step=256):
                path_c_float32_abi_bank[359066797 + ((local_gb10_quarter_brick_10_M_bwd_grad_flat))] = path_c_float32_abi_bank[359066797 + ((local_gb10_quarter_brick_10_M_bwd_grad_flat))] + (path_c_float32_abi_bank[row * 3584 + (local_gb10_quarter_brick_10_M_bwd_grad_flat // 7168)] * local_gb10_quarter_brick_10_M_delta_grad[(row * 3584 + (local_gb10_quarter_brick_10_M_bwd_grad_flat // 7168)) % 3584])
            T.sync_threads()
            for local_gb10_quarter_brick_10_M_bwd_state_idx in T.serial(lane, 458752, step=256):
                path_c_float32_abi_bank[384810381 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 458752)] = path_c_float32_abi_bank[384810381 + ((local_gb10_quarter_brick_10_M_bwd_state_idx) % 458752)] + local_gb10_quarter_brick_10_M_delta_grad[(row * 3584 + (local_gb10_quarter_brick_10_M_bwd_state_idx % 3584)) % 3584]
            T.sync_threads()
