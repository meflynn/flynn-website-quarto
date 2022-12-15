// generated with brms 2.18.0
functions {
  /* integer sequence of values
   * Args:
   *   start: starting integer
   *   end: ending integer
   * Returns:
   *   an integer sequence from start to end
   */
  array[] int sequence(int start, int end) {
    array[end - start + 1] int seq;
    for (n in 1 : num_elements(seq)) {
      seq[n] = n + start - 1;
    }
    return seq;
  }
  // compute partial sums of the log-likelihood
  real partial_log_lik_lpmf(array[] int seq, int start, int end,
                            data int ncat, data array[] int Y,
                            vector bQ_mudk, real Intercept_mudk,
                            data matrix XQ_mudk, vector bQ_muneg,
                            real Intercept_muneg, data matrix XQ_muneg,
                            vector bQ_mupos, real Intercept_mupos,
                            data matrix XQ_mupos, data array[] int J_1,
                            data vector Z_1_mudk_1, vector r_1_mudk_1,
                            data array[] int J_2, data vector Z_2_muneg_1,
                            vector r_2_muneg_1, data array[] int J_3,
                            data vector Z_3_mupos_1, vector r_3_mupos_1) {
    real ptarget = 0;
    int N = end - start + 1;
    // initialize linear predictor term
    vector[N] mudk = rep_vector(0.0, N);
    // initialize linear predictor term
    vector[N] muneg = rep_vector(0.0, N);
    // initialize linear predictor term
    vector[N] mupos = rep_vector(0.0, N);
    // linear predictor matrix
    array[N] vector[ncat] mu;
    mudk += Intercept_mudk + XQ_mudk[start : end] * bQ_mudk;
    muneg += Intercept_muneg + XQ_muneg[start : end] * bQ_muneg;
    mupos += Intercept_mupos + XQ_mupos[start : end] * bQ_mupos;
    for (n in 1 : N) {
      // add more terms to the linear predictor
      int nn = n + start - 1;
      mudk[n] += r_1_mudk_1[J_1[nn]] * Z_1_mudk_1[nn];
    }
    for (n in 1 : N) {
      // add more terms to the linear predictor
      int nn = n + start - 1;
      muneg[n] += r_2_muneg_1[J_2[nn]] * Z_2_muneg_1[nn];
    }
    for (n in 1 : N) {
      // add more terms to the linear predictor
      int nn = n + start - 1;
      mupos[n] += r_3_mupos_1[J_3[nn]] * Z_3_mupos_1[nn];
    }
    for (n in 1 : N) {
      mu[n] = transpose([mudk[n], muneg[n], 0, mupos[n]]);
    }
    for (n in 1 : N) {
      int nn = n + start - 1;
      ptarget += categorical_logit_lpmf(Y[nn] | mu[n]);
    }
    return ptarget;
  }
}
data {
  int<lower=1> N; // total number of observations
  int<lower=2> ncat; // number of categories
  array[N] int Y; // response variable
  int<lower=1> K_mudk; // number of population-level effects
  matrix[N, K_mudk] X_mudk; // population-level design matrix
  int<lower=1> K_muneg; // number of population-level effects
  matrix[N, K_muneg] X_muneg; // population-level design matrix
  int<lower=1> K_mupos; // number of population-level effects
  matrix[N, K_mupos] X_mupos; // population-level design matrix
  int grainsize; // grainsize for threading
  // data for group-level effects of ID 1
  int<lower=1> N_1; // number of grouping levels
  int<lower=1> M_1; // number of coefficients per level
  array[N] int<lower=1> J_1; // grouping indicator per observation
  // group-level predictor values
  vector[N] Z_1_mudk_1;
  // data for group-level effects of ID 2
  int<lower=1> N_2; // number of grouping levels
  int<lower=1> M_2; // number of coefficients per level
  array[N] int<lower=1> J_2; // grouping indicator per observation
  // group-level predictor values
  vector[N] Z_2_muneg_1;
  // data for group-level effects of ID 3
  int<lower=1> N_3; // number of grouping levels
  int<lower=1> M_3; // number of coefficients per level
  array[N] int<lower=1> J_3; // grouping indicator per observation
  // group-level predictor values
  vector[N] Z_3_mupos_1;
  int prior_only; // should the likelihood be ignored?
}
transformed data {
  int Kc_mudk = K_mudk - 1;
  matrix[N, Kc_mudk] Xc_mudk; // centered version of X_mudk without an intercept
  vector[Kc_mudk] means_X_mudk; // column means of X_mudk before centering
  // matrices for QR decomposition
  matrix[N, Kc_mudk] XQ_mudk;
  matrix[Kc_mudk, Kc_mudk] XR_mudk;
  matrix[Kc_mudk, Kc_mudk] XR_mudk_inv;
  int Kc_muneg = K_muneg - 1;
  matrix[N, Kc_muneg] Xc_muneg; // centered version of X_muneg without an intercept
  vector[Kc_muneg] means_X_muneg; // column means of X_muneg before centering
  // matrices for QR decomposition
  matrix[N, Kc_muneg] XQ_muneg;
  matrix[Kc_muneg, Kc_muneg] XR_muneg;
  matrix[Kc_muneg, Kc_muneg] XR_muneg_inv;
  int Kc_mupos = K_mupos - 1;
  matrix[N, Kc_mupos] Xc_mupos; // centered version of X_mupos without an intercept
  vector[Kc_mupos] means_X_mupos; // column means of X_mupos before centering
  // matrices for QR decomposition
  matrix[N, Kc_mupos] XQ_mupos;
  matrix[Kc_mupos, Kc_mupos] XR_mupos;
  matrix[Kc_mupos, Kc_mupos] XR_mupos_inv;
  array[N] int seq = sequence(1, N);
  for (i in 2 : K_mudk) {
    means_X_mudk[i - 1] = mean(X_mudk[ : , i]);
    Xc_mudk[ : , i - 1] = X_mudk[ : , i] - means_X_mudk[i - 1];
  }
  // compute and scale QR decomposition
  XQ_mudk = qr_thin_Q(Xc_mudk) * sqrt(N - 1);
  XR_mudk = qr_thin_R(Xc_mudk) / sqrt(N - 1);
  XR_mudk_inv = inverse(XR_mudk);
  for (i in 2 : K_muneg) {
    means_X_muneg[i - 1] = mean(X_muneg[ : , i]);
    Xc_muneg[ : , i - 1] = X_muneg[ : , i] - means_X_muneg[i - 1];
  }
  // compute and scale QR decomposition
  XQ_muneg = qr_thin_Q(Xc_muneg) * sqrt(N - 1);
  XR_muneg = qr_thin_R(Xc_muneg) / sqrt(N - 1);
  XR_muneg_inv = inverse(XR_muneg);
  for (i in 2 : K_mupos) {
    means_X_mupos[i - 1] = mean(X_mupos[ : , i]);
    Xc_mupos[ : , i - 1] = X_mupos[ : , i] - means_X_mupos[i - 1];
  }
  // compute and scale QR decomposition
  XQ_mupos = qr_thin_Q(Xc_mupos) * sqrt(N - 1);
  XR_mupos = qr_thin_R(Xc_mupos) / sqrt(N - 1);
  XR_mupos_inv = inverse(XR_mupos);
}
parameters {
  vector[Kc_mudk] bQ_mudk; // regression coefficients at QR scale
  real Intercept_mudk; // temporary intercept for centered predictors
  vector[Kc_muneg] bQ_muneg; // regression coefficients at QR scale
  real Intercept_muneg; // temporary intercept for centered predictors
  vector[Kc_mupos] bQ_mupos; // regression coefficients at QR scale
  real Intercept_mupos; // temporary intercept for centered predictors
  vector<lower=0>[M_1] sd_1; // group-level standard deviations
  array[M_1] vector[N_1] z_1; // standardized group-level effects
  vector<lower=0>[M_2] sd_2; // group-level standard deviations
  array[M_2] vector[N_2] z_2; // standardized group-level effects
  vector<lower=0>[M_3] sd_3; // group-level standard deviations
  array[M_3] vector[N_3] z_3; // standardized group-level effects
}
transformed parameters {
  vector[N_1] r_1_mudk_1; // actual group-level effects
  vector[N_2] r_2_muneg_1; // actual group-level effects
  vector[N_3] r_3_mupos_1; // actual group-level effects
  real lprior = 0; // prior contributions to the log posterior
  r_1_mudk_1 = sd_1[1] * z_1[1];
  r_2_muneg_1 = sd_2[1] * z_2[1];
  r_3_mupos_1 = sd_3[1] * z_3[1];
  lprior += normal_lpdf(bQ_mudk | 0, 1);
  lprior += student_t_lpdf(Intercept_mudk | 3, 0, 2.5);
  lprior += normal_lpdf(bQ_muneg | 0, 1);
  lprior += student_t_lpdf(Intercept_muneg | 3, 0, 2.5);
  lprior += normal_lpdf(bQ_mupos | 0, 1);
  lprior += student_t_lpdf(Intercept_mupos | 3, 0, 2.5);
  lprior += student_t_lpdf(sd_1 | 3, 0, 2.5)
            - 1 * student_t_lccdf(0 | 3, 0, 2.5);
  lprior += student_t_lpdf(sd_2 | 3, 0, 2.5)
            - 1 * student_t_lccdf(0 | 3, 0, 2.5);
  lprior += student_t_lpdf(sd_3 | 3, 0, 2.5)
            - 1 * student_t_lccdf(0 | 3, 0, 2.5);
}
model {
  // likelihood including constants
  if (!prior_only) {
    target += reduce_sum(partial_log_lik_lpmf, seq, grainsize, ncat, Y,
                         bQ_mudk, Intercept_mudk, XQ_mudk, bQ_muneg,
                         Intercept_muneg, XQ_muneg, bQ_mupos,
                         Intercept_mupos, XQ_mupos, J_1, Z_1_mudk_1,
                         r_1_mudk_1, J_2, Z_2_muneg_1, r_2_muneg_1, J_3,
                         Z_3_mupos_1, r_3_mupos_1);
  }
  // priors including constants
  target += lprior;
  target += std_normal_lpdf(z_1[1]);
  target += std_normal_lpdf(z_2[1]);
  target += std_normal_lpdf(z_3[1]);
}
generated quantities {
  // obtain the actual coefficients
  vector[Kc_mudk] b_mudk = XR_mudk_inv * bQ_mudk;
  // actual population-level intercept
  real b_mudk_Intercept = Intercept_mudk - dot_product(means_X_mudk, b_mudk);
  // obtain the actual coefficients
  vector[Kc_muneg] b_muneg = XR_muneg_inv * bQ_muneg;
  // actual population-level intercept
  real b_muneg_Intercept = Intercept_muneg
                           - dot_product(means_X_muneg, b_muneg);
  // obtain the actual coefficients
  vector[Kc_mupos] b_mupos = XR_mupos_inv * bQ_mupos;
  // actual population-level intercept
  real b_mupos_Intercept = Intercept_mupos
                           - dot_product(means_X_mupos, b_mupos);
}

