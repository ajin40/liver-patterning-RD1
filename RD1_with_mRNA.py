"""
implementation of RD1
"""

protein_degradation_rate = 0.125
translation_rate = 0.167

def d_mRNA_A(mRNA, A, I, ind1, ind2, Km_aa, Km_ai, Km_am, a0, a1, a2, a3, a4, n, m, p):
    """
    mRNA Activator has induced expression + self-activation + inhibition by inhibitor
    :param A: concentration of inducer
    :param n: hill coefficient
    """
    return a0 * ind1 + (a1 * A ** n)/(Km_aa ** n + A ** n) * (a2 * Km_ai ** m)/(Km_ai ** m + I ** m) * (a3 * ind2 ** p)/(Km_am ** p + ind2 ** p) - a4 * mRNA

def d_A(mRNA_A, A, alpha=translation_rate, beta=protein_degradation_rate):
    return mRNA_A * alpha - A * beta

def d_mRNA_I(mRNA, A, I, ind2, Km_ia, Km_ii, Km_im, b0, b1, b2, b3, b4, n, m, p):
    """
    Inhibitor activated by activator + self-inhibition + increased activation by modulator
    :param I: concentration of inhibitor
    :param n: hill coefficient
    """
    return b0 + (b1 * A ** n)/(Km_ia ** n + A ** n) * (b2 * Km_ii**m)/(Km_ii ** m + I ** m) * (b3 * ind2 ** p)/(Km_im ** p + ind2 ** p) - b4 * mRNA

def d_I(mRNA_I, I, alpha=translation_rate, beta=protein_degradation_rate):
    return mRNA_I * alpha - I * beta

def dudt(mRNA_A, mRNA_I, mRNA_M, A, I, M, active, dt, cell_types,
         mRNA_transcription, mRNA_degradation, protein_translation, protein_degradation,
         hill_km, hill_n, hill_interaction_parameters):
    """
    Solving a system of equations of A and I for inducer concentration ind2
    :param A: current concentration of activator
    :param I: current concentration of inhibitor
    :param dt: timestep for ode interval
    :param ind1: concentration of doxycycline (constant)
    :param ind2: concentration of modulator
    """
    a0, b0, c0 = mRNA_transcription
    a_deg, b_deg, c_deg = mRNA_degradation
    a_alpha, b_alpha, c_alpha = protein_translation
    a_beta, b_beta, c_beta = protein_degradation

    #interaction terms
    a1, b1, a2, a3 = hill_interaction_parameters
    Km_a, Km_i, Km_m = hill_km
    n, m, p = hill_n

    dmRNA_A_dt = d_mRNA_A(mRNA_A, A, I, active, M, Km_a, Km_i, Km_m, a0, a1, a2, a3, a_deg, n, m, p) * dt * (cell_types != 3)
    dmRNA_I_dt = d_mRNA_I(mRNA_I, A, I, M, Km_a, Km_i, Km_m, b0, b1, a2, a3, b_deg, n, m, p) * dt * (cell_types != 3)
    dmRNA_M_dt = d_mRNA_M(mRNA_M, cell_types, active, c0, c_deg) * dt * (cell_types == 3)
    dA_dt = d_A(mRNA_A, A, a_alpha, a_beta) * dt
    dI_dt = d_I(mRNA_I, I, b_alpha, b_beta) * dt
    dM_dt = d_M(mRNA_M, M, c_alpha, c_beta) * dt
    return mRNA_A + dmRNA_A_dt, mRNA_I + dmRNA_I_dt, mRNA_M + dmRNA_M_dt, A + dA_dt, I + dI_dt, M + dM_dt


def d_mRNA_M(mRNA, cell_type, ind1, r_transcription_mRNA_M, r_degradation_mRNA_M):
    """
    Modulator regulates inhibition
    """
    #r_transcription_mRNA_M * (cell_type == 3) * ind1 - r_degradation_mRNA_M * mRNA
    return r_transcription_mRNA_M * ind1 - r_degradation_mRNA_M * mRNA

def d_M(mRNA_M, M, alpha=translation_rate, beta=protein_degradation_rate):
    return alpha * mRNA_M - beta * M
