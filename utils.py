import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def drift(z, alpha, beta):
    return -z**3 + beta * z + alpha

def euler(alpha, beta, sigma, z0, T, dt):
    n_steps = int(T / dt)
    z = np.zeros(n_steps)
    z[0] = z0
    for t in range(1, n_steps):
        dW = np.sqrt(dt) * np.random.normal(0, 1)
        z[t] = z[t-1] + drift(z[t-1], alpha[t], beta[t]) * dt + sigma * dW
    return z

def euler_sim(alpha, beta, sigma, z0, T, dt):
    n_steps = int(T / dt)
    z = np.zeros(n_steps)
    z[0] = z0
    for t in range(1, n_steps):
        dW = np.sqrt(dt) * np.random.normal(0, 1) 
        z[t] = z[t-1] + drift(z[t-1], alpha, beta) * dt + sigma * dW
    return z

def simulate_system(alpha, beta, sigma, z0, T, dt):
    z_values = euler_sim(alpha, beta, sigma, z0, T, dt)
    return z_values

def negative_log_likelihood_sim(params, data, z0, T, dt):
    alpha, beta, sigma = params
    simulated_data = euler_sim(alpha, beta, sigma, z0, T, dt)
    rss = np.sum((simulated_data - data) ** 2)
    return rss

def negative_log_likelihood(params, data, features, z0, T, dt):
    alpha0, alpha1, alpha2, alpha3, beta0, beta1, beta2, sigma = params
    alpha = alpha0 + alpha1 * features['interest_rate_scaled'] + alpha2 * features['house_age_scaled'] + alpha3 * features['distance_from_mrt_scaled']
    beta = beta0 + beta1 * features['interest_rate_scaled'] + beta2 * features['n_of_conv_stores_scaled']
    z_ = euler(alpha, beta, sigma, z0, T, dt)
    if np.any(np.isnan(z_)) or np.any(np.isinf(z_)):
        return np.inf
    rss = np.sum((data['price_scaled'] - z_) ** 2)
    return rss

def negative_log_likelihood_cobb(params, data, features):
    alpha0, alpha1, alpha2, alpha3, beta0, beta1, beta2 = params
    alpha = alpha0 + alpha1 * features['interest_rate_scaled'] + alpha2 * features['house_age_scaled'] + alpha3 * features['distance_from_mrt_scaled']
    beta = beta0 + beta1 * features['interest_rate_scaled'] + beta2 * features['n_of_conv_stores_scaled']
    likelihoods = []
    for i in range(len(data)):
        z = data['price_scaled'].iloc[i]
        likelihood = cusp_pdf(z, alpha.iloc[i], beta.iloc[i])
        if likelihood > 0:
            likelihoods.append(-np.log(likelihood))
        else:
            likelihoods.append(np.inf)
    return np.sum(likelihoods)

def cusp_pdf(z, alpha, beta):
    exponent = alpha * z + (beta * z**2) / 2 - (z**4) / 4
    exponent = np.clip(exponent, -100, 100)
    return np.exp(exponent)

def predict_prices(alpha_series, beta_series):
    predicted_prices = []
    for alpha, beta in zip(alpha_series, beta_series):
        predicted_price = minimize(lambda z: -cusp_pdf(z, alpha, beta), 0).x[0]
        predicted_prices.append(predicted_price)
    return np.array(predicted_prices)

def normalize_pdf(alpha, beta):
    z_values = np.linspace(-5, 5, 1000)
    pdf_values = cusp_pdf(z_values, alpha, beta)
    return np.trapz(pdf_values, z_values)

def cobb_simulate(alpha, beta, n_samples):
    z_samples = []
    for _ in range(n_samples):
        while True:
            z = np.random.uniform(-5, 5)
            pdf_value = cusp_pdf(z, alpha, beta)
            if np.random.uniform(0, 1) < pdf_value / normalize_pdf(alpha, beta):
                z_samples.append(z)
                break
    return np.array(z_samples)

def negative_log_likelihood_cobb_sim(params, data):
    alpha, beta, sigma = params
    likelihoods = []
    for z in data:
        pdf_value = cusp_pdf(z, alpha, beta)
        if pdf_value > 0:
            likelihoods.append(-np.log(pdf_value))
        else:
            likelihoods.append(np.inf)
    return np.sum(likelihoods)
