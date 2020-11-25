def crra_concavity(c, sigma=0.4):
    '''This function returns the value of utility when the CRRA
    coefficient is sigma. I.e. 
    u(c,sigma)=(c**(1-sigma)-1)/(1-sigma) if sigma!=1 
    and 
    u(c,sigma)=ln(c) if sigma==1
    Usage: u(c,sigma)
    '''
    if sigma!=1:
        u = (c**(1-sigma) - 1) / (1-sigma)
    else:
        u = np.log(c)
    return u
