# here our simulations are waveforms with parameter only m1, m2 rest we keep same
# can we recover m1, m2
import torch
import numpy as np
import torch
import sbi.utils as utils
from sbi.inference.base import infer
from sbi import analysis as analysis
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform



def get_22mode(m2, m1, approx='IMRPhenomXPHM'):
    """
    Not m2 >= m1 as q=m1/m2 is >=1

    can get many modes but lets just return 22 mode
    use mode_array for other modes
    """
    hp, hc = get_td_waveform(approximant=approx,
                         mass1=m1,
                         mass2=m2,
                         f_lower=20.0,
                         mode_array=[(2,2)],
                         inclination = 1.0,
                         delta_t=1.0/4096)
    #hp, hc = hp.trim_zeros(), hc.trim_zeros()

    a = hp.max()
    return hp / a

#h22s = get_22mode(10, 40, approx='SpinTaylorT4')
#t = h22s.sample_times
#
#time = np.array(t)
#print(h22s)
#h22s.plot(label='2,2')
##plt.xlim(-1, 0.05)
#plt.plot(time, h22s.real())
#plt.plot(time, h22s.imag())
#plt.legend()
#plt.xlabel('Time [s]')
#plt.ylabel('Relative Strain')
#plt.show()
#

#### range of m1 and m2
prior = utils.BoxUniform(
    low=torch.tensor([ 5., 100.]),
    high=torch.tensor([2., 100.])
)


# Here we need to get waveforms given m1, m2
def simulator(mu):
    # Generate samples [waveforms]
    return get_22mode(mu[0], mu[1]) #mu + 0.5 * torch.randn_like(mu)

num_sim = 20
method = 'SNRE' #SNPE or SNLE or SNRE
posterior = infer(
    simulator,
    prior,
    # See glossary for explanation of methods.
    #    SNRE newer than SNLE newer than SNPE.
    method=method,
    num_workers=-1,
    num_simulations=num_sim)

n_observations = 3
observation = torch.tensor([10, 1.5])[None] + 0.5*torch.randn(n_observations, 2)

from matplotlib import pyplot as plt
plt.figure()
plt.scatter(x=observation[:, 0], y=observation[:, 1])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim(2, 100)
plt.ylim(2, 100)
plt.show()
quit()
samples = posterior.sample((200,), x=observation[0])

log_probability = posterior.log_prob(samples, x=observation[0])
_ = analysis.pairplot(samples, limits=[[-5,5],[-5,5]], fig_size=(6,6), upper='kde', diag='kde')
plt.show()
print(posterior)
import numpy as np

bounds = [3-1, 3+1, -1.5-1, -1.5+1]

mu_1, mu_2 = torch.tensor(np.mgrid[bounds[0]:bounds[1]:2/50., bounds[2]:bounds[3]:2/50.]).float()

grids = torch.cat(
    (mu_1.reshape(-1, 1), mu_2.reshape(-1, 1)),
    dim=1
)

#if method == 'SNPE':
log_prob = sum([
        posterior.log_prob(grids, observation[i])
        for i in range(len(observation))
    ])
#else:
#    log_prob = sum([
#        posterior.set_default_x(torch.cat((grids, observation[i].repeat((grids.shape[0])).reshape(-1, 2)), dim=1))[:, 0] + posterior._prior.log_prob(grids)
#        for i in range(len(observation))
#    ]).detach()

prob = torch.exp(log_prob - log_prob.max())
plt.figure(dpi=200)
plt.plot([2, 4], [-1.5, -1.5], color='k')
plt.plot([3, 3], [-0.5, -2.5], color='k')
plt.contourf(prob.reshape(*mu_1.shape), extent=bounds, origin='lower')
plt.axis('scaled')
plt.xlim(2+0.3, 4-0.3)
plt.ylim(-2.5+0.3, -0.5-0.3)
plt.title('Posterior with learned likelihood\nfrom %d examples of'%(num_sim)+r' $\mu_i\in[-5, 5]$')
plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')

plt.show()


true_like = lambda x: -((x[0] - mu_1)**2 + (x[1] - mu_2)**2)/(2*0.5**2)
log_prob = sum([true_like(observation[i]) for i in range(len(observation))])
plt.figure(dpi=200)
prob = torch.exp(log_prob - log_prob.max())
plt.plot([2, 4], [-1.5, -1.5], color='k')
plt.plot([3, 3], [-0.5, -2.5], color='k')
plt.contourf(prob.reshape(*mu_1.shape), extent=bounds, origin='lower')
plt.axis('scaled')
plt.xlim(2+0.3, 4-0.3)
plt.ylim(-2.5+0.3, -0.5-0.3)
plt.title('Posterior with\nanalytic likelihood')
plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')

plt.title("True Distribution")
plt.show()
