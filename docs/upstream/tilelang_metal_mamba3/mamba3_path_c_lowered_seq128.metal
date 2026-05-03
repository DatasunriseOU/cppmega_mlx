// === Path C (TileLang DSL) lowered MSL ===
// Bench shape: B=2 T=128 H=4 P=32 N=64

// ---- Forward ----
// Function: fwd_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void fwd_kernel(  device float* A [[ buffer(0) ]],
  device float* B [[ buffer(1) ]],
  device float* C [[ buffer(2) ]],
  device float* D [[ buffer(3) ]],
  device float* dt [[ buffer(4) ]],
  device float* h0 [[ buffer(5) ]],
  device float* h_last [[ buffer(6) ]],
  device float* x [[ buffer(7) ]],
  device float* y [[ buffer(8) ]],
  device float* z [[ buffer(9) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]]
) {
  thread float h_state[64];
  thread float y_acc[1];
  for (int n = 0; n < 64; ++n) {
    h_state[n] = h0[((((int)threadIdx.x) * 64) + n)];
  }
  for (int t = 0; t < 128; ++t) {
    float A_val = A[((((((int)threadIdx.x) >> 7) * 512) + (t * 4)) + ((((int)threadIdx.x) & 127) >> 5))];
    float dt_val = dt[((((((int)threadIdx.x) >> 7) * 512) + (t * 4)) + ((((int)threadIdx.x) & 127) >> 5))];
    float decay = exp((A_val * dt_val));
    float x_val = x[((((((int)threadIdx.x) >> 7) * 16384) + (t * 128)) + (((int)threadIdx.x) & 127))];
    float z_val = z[((((((int)threadIdx.x) >> 7) * 16384) + (t * 128)) + (((int)threadIdx.x) & 127))];
    y_acc[0] = 0.000000e+00f;
    for (int n_1 = 0; n_1 < 64; ++n_1) {
      float new_h = ((decay * h_state[n_1]) + (x_val * B[(((((((int)threadIdx.x) >> 7) * 32768) + (t * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + n_1)]));
      h_state[n_1] = new_h;
      y_acc[0] = (y_acc[0] + (new_h * C[(((((((int)threadIdx.x) >> 7) * 32768) + (t * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + n_1)]));
    }
    float D_h = D[((((int)threadIdx.x) & 127) >> 5)];
    float y_skipped = (y_acc[0] + (D_h * x_val));
    float sig_z = (1.000000e+00f / (1.000000e+00f + exp((z_val * -1.000000e+00f))));
    y[((((((int)threadIdx.x) >> 7) * 16384) + (t * 128)) + (((int)threadIdx.x) & 127))] = ((z_val * sig_z) * y_skipped);
  }
  for (int n_2 = 0; n_2 < 64; ++n_2) {
    h_last[((((int)threadIdx.x) * 64) + n_2)] = h_state[n_2];
  }
}



// ---- Backward ----
// Function: bwd_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void bwd_kernel(  device float* A [[ buffer(0) ]],
  device float* B [[ buffer(1) ]],
  device float* C [[ buffer(2) ]],
  device float* D [[ buffer(3) ]],
  device float* dA_partial [[ buffer(4) ]],
  device float* dB_partial [[ buffer(5) ]],
  device float* dC_partial [[ buffer(6) ]],
  device float* dD_partial [[ buffer(7) ]],
  device float* ddt_partial [[ buffer(8) ]],
  device float* dh0 [[ buffer(9) ]],
  device float* dt [[ buffer(10) ]],
  device float* dx [[ buffer(11) ]],
  device float* dy [[ buffer(12) ]],
  device float* dz [[ buffer(13) ]],
  device float* h0 [[ buffer(14) ]],
  device float* h_steps [[ buffer(15) ]],
  device float* x [[ buffer(16) ]],
  device float* z [[ buffer(17) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]]
) {
  thread float h_state[64];
  thread float dh[64];
  thread float dD_acc[1];
  thread float y_state[1];
  thread float dx_inp[1];
  thread float d_decay[1];
  for (int n = 0; n < 64; ++n) {
    h_state[n] = h0[((((int)threadIdx.x) * 64) + n)];
  }
  for (int t = 0; t < 128; ++t) {
    float A_val = A[((((((int)threadIdx.x) >> 7) * 512) + (t * 4)) + ((((int)threadIdx.x) & 127) >> 5))];
    float dt_val = dt[((((((int)threadIdx.x) >> 7) * 512) + (t * 4)) + ((((int)threadIdx.x) & 127) >> 5))];
    float decay = exp((A_val * dt_val));
    float x_val = x[((((((int)threadIdx.x) >> 7) * 16384) + (t * 128)) + (((int)threadIdx.x) & 127))];
    for (int n_1 = 0; n_1 < 64; ++n_1) {
      float new_h = ((decay * h_state[n_1]) + (x_val * B[(((((((int)threadIdx.x) >> 7) * 32768) + (t * 256)) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + n_1)]));
      h_state[n_1] = new_h;
      h_steps[(((((int)threadIdx.x) * 8192) + (t * 64)) + n_1)] = new_h;
    }
  }
  for (int n_2 = 0; n_2 < 64; ++n_2) {
    dh[n_2] = 0.000000e+00f;
  }
  dD_acc[0] = 0.000000e+00f;
  float D_h = D[((((int)threadIdx.x) & 127) >> 5)];
  for (int r = 0; r < 128; ++r) {
    float A_val_1 = A[(((((((int)threadIdx.x) >> 7) * 512) + ((((int)threadIdx.x) & 127) >> 5)) + 508) - (r * 4))];
    float dt_val_1 = dt[(((((((int)threadIdx.x) >> 7) * 512) + ((((int)threadIdx.x) & 127) >> 5)) + 508) - (r * 4))];
    float decay_1 = exp((A_val_1 * dt_val_1));
    float x_val_1 = x[(((((((int)threadIdx.x) >> 7) * 16384) + (((int)threadIdx.x) & 127)) + 16256) - (r * 128))];
    float z_val = z[(((((((int)threadIdx.x) >> 7) * 16384) + (((int)threadIdx.x) & 127)) + 16256) - (r * 128))];
    float dY = dy[(((((((int)threadIdx.x) >> 7) * 16384) + (((int)threadIdx.x) & 127)) + 16256) - (r * 128))];
    y_state[0] = 0.000000e+00f;
    for (int n_3 = 0; n_3 < 64; ++n_3) {
      y_state[0] = (y_state[0] + (h_steps[((((((int)threadIdx.x) * 8192) + n_3) + 8128) - (r * 64))] * C[((((((((int)threadIdx.x) >> 7) * 32768) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + n_3) + 32512) - (r * 256))]));
    }
    float y_skipped = (y_state[0] + (D_h * x_val_1));
    float sig_z = (1.000000e+00f / (1.000000e+00f + exp((z_val * -1.000000e+00f))));
    float silu_z = (z_val * sig_z);
    float silu_dz = (sig_z * (1.000000e+00f + (z_val * (1.000000e+00f - sig_z))));
    float d_silu = (dY * y_skipped);
    float d_y_skipped = (dY * silu_z);
    dz[(((((((int)threadIdx.x) >> 7) * 16384) + (((int)threadIdx.x) & 127)) + 16256) - (r * 128))] = (d_silu * silu_dz);
    dD_acc[0] = (dD_acc[0] + (d_y_skipped * x_val_1));
    for (int n_4 = 0; n_4 < 64; ++n_4) {
      dh[n_4] = (dh[n_4] + (d_y_skipped * C[((((((((int)threadIdx.x) >> 7) * 32768) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + n_4) + 32512) - (r * 256))]));
    }
    for (int n_5 = 0; n_5 < 64; ++n_5) {
      dC_partial[((((((((int)threadIdx.x) >> 7) * 1048576) + ((((int)threadIdx.x) & 127) * 64)) + n_5) + 1040384) - (r * 8192))] = (d_y_skipped * h_steps[((((((int)threadIdx.x) * 8192) + n_5) + 8128) - (r * 64))]);
      dB_partial[((((((((int)threadIdx.x) >> 7) * 1048576) + ((((int)threadIdx.x) & 127) * 64)) + n_5) + 1040384) - (r * 8192))] = (dh[n_5] * x_val_1);
    }
    dx_inp[0] = 0.000000e+00f;
    for (int n_6 = 0; n_6 < 64; ++n_6) {
      dx_inp[0] = (dx_inp[0] + (dh[n_6] * B[((((((((int)threadIdx.x) >> 7) * 32768) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + n_6) + 32512) - (r * 256))]));
    }
    float dx_skip = (d_y_skipped * D_h);
    dx[(((((((int)threadIdx.x) >> 7) * 16384) + (((int)threadIdx.x) & 127)) + 16256) - (r * 128))] = (dx_skip + dx_inp[0]);
    d_decay[0] = 0.000000e+00f;
    if (r == 127) {
      for (int n_7 = 0; n_7 < 64; ++n_7) {
        d_decay[0] = (d_decay[0] + (dh[n_7] * h0[((((int)threadIdx.x) * 64) + n_7)]));
      }
    } else {
      for (int n_8 = 0; n_8 < 64; ++n_8) {
        d_decay[0] = (d_decay[0] + (dh[n_8] * h_steps[((((((int)threadIdx.x) * 8192) + n_8) + 8064) - (r * 64))]));
      }
    }
    float d_logdecay = (d_decay[0] * decay_1);
    dA_partial[(((((((int)threadIdx.x) >> 7) * 16384) + (((int)threadIdx.x) & 127)) + 16256) - (r * 128))] = (d_logdecay * dt_val_1);
    ddt_partial[(((((((int)threadIdx.x) >> 7) * 16384) + (((int)threadIdx.x) & 127)) + 16256) - (r * 128))] = (d_logdecay * A_val_1);
    for (int n_9 = 0; n_9 < 64; ++n_9) {
      dh[n_9] = (dh[n_9] * decay_1);
    }
  }
  for (int n_10 = 0; n_10 < 64; ++n_10) {
    dh0[((((int)threadIdx.x) * 64) + n_10)] = dh[n_10];
  }
  dD_partial[((int)threadIdx.x)] = dD_acc[0];
}


