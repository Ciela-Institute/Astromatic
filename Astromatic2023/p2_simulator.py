import pyro
import torch
from caustic import Simulator as Base, EPL, Sersic, FlatLambdaCDM, get_meshgrid
from pyro import distributions as dist
from collections import OrderedDict

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

class Simulator(Base):
    def __init__(self, z_l=0.5, z_s=1.0, n_pix=100, pixel_scale=0.05, with_lens_light=True):
        """
        z_l: redshift of the lens
        z_s: redshift of the source
        n_pix: Number of pixels in the observation
        pixel_scale: Size of the pixels in the observation, in arcseconds.
        """
        super().__init__()
        self.with_lens_light = with_lens_light
        self.cosmo = FlatLambdaCDM(name="Cosmology")
        self.lens = EPL(self.cosmo, z_l=z_l, name="Lens")
        self.source = Sersic(name="Source")
        if self.with_lens_light:
            self.lens_light = Sersic(name="Lens_light")
        
        self.thx, self.thy = get_meshgrid(pixel_scale, n_pix, n_pix)
        self.n_pix = n_pix
        self.add_param("z_s", z_s)

    def forward(self, params):
        z_s = self.unpack(params)
        alphax, alphay = self.lens.reduced_deflection_angle(x=self.thx, y=self.thy, z_s=z_s, params=params) 
        bx = self.thx - alphax
        by = self.thy - alphay
        image = self.source.brightness(bx, by, params)
        if self.with_lens_light:
            image += self.lens_light.brightness(self.thx, self.thy, params)
        return image
    
    
def prior(N, simulator, prior_params): # N is the number of samples we want to draw from the prior
    """
    We iterate over the "dynamic" parameters in our simulator 
    and collect them in a dictionary to be fed in the simulator
    """
    pyro.clear_param_store()
    
    with pyro.plate("N", N, device=DEVICE):
        params = OrderedDict(name: OrderedDict() for name in  simulator.params.dynamic.keys())
        for name, module in simulator.params.dynamic.items():
            for p in module.keys():
                if p=="x0" or p=="y0": # positional parameter
                    _mean = prior_params["positional"]["mean"]
                    _sigma = prior_params["positional"]["sigma"]
                    mean = pyro.param(f"{name}_{p}_mu", torch.tensor(_mean).float())
                    sigma = pyro.param(f"{name}_{p}_sigma", torch.tensor(_sigma).float())
                    if name == "Lens":
                        v = pyro.sample(f"Lens_{p}", dist.Normal(mean, sigma)) # link the lens light axis ratio to lens
                        params[name].update({p: v})
                        if "Lens_light" in params.keys():
                            params["Lens_light"].update({p: v})
                    elif name != "Lens_light":
                        v = pyro.sample(f"{name}_{p}", dist.Normal(mean, sigma))
                        params[name].update({p: v})
                elif p=="q": # Axis ratio
                    _min = prior_params["axis_ratio"]["min"]
                    _max = prior_params["axis_ratio"]["max"]
                    min_value = pyro.param(f"{name}_{p}_min", torch.tensor(_min).float())
                    max_value = pyro.param(f"{name}_{p}_max", torch.tensor(_max).float())
                    if name == "Lens":
                        v = pyro.sample(f"Lens_{p}", dist.Uniform(min_value, max_value)) # link the lens light axis ratio to lens
                        params[name].update({p: v})
                        if "Lens_light" in params.keys():
                            params["Lens_light"].update({p: v})
                    elif name != "Lens_light":
                        v = pyro.sample(f"{name}_{p}", dist.Uniform(min_value, max_value))
                        params[name].update({p: v})
                elif p=="phi": # Orientation
                    _min = prior_params["orientation"]["min"]
                    _max = prior_params["orientation"]["max"]
                    min_value = pyro.param(f"{name}_{p}_min", torch.tensor(_min).float())
                    max_value = pyro.param(f"{name}_{p}_max", torch.tensor(_max).float()) 
                    if name == "Lens":
                        v = pyro.sample(f"Lens_{p}", dist.Uniform(min_value, max_value)) # link the lens light orientation to lens
                        params[name].update({p: v})
                        if "Lens_light" in params.keys():
                            params["Lens_light"].update({p: v})
                    elif name != "Lens_light":
                        v = pyro.sample(f"{name}_{p}", dist.Uniform(min_value, max_value))
                        params[name].update({p: v})
                elif p=="n": # Sersic index
                    _min = prior_params["sersic_index"]["min"]
                    _max = prior_params["sersic_index"]["max"]
                    min_value = pyro.param(f"{name}_{p}_min", torch.tensor(_min).float())
                    max_value = pyro.param(f"{name}_{p}_max", torch.tensor(_max).float()) 
                    v = pyro.sample(f"{name}_{p}", dist.Uniform(min_value, max_value))
                    params[name].update({p: v})
                elif p=="Re": # Sersic effective radius
                    _min = prior_params["sersic_effective_radius"]["min"]
                    _max = prior_params["sersic_effective_radius"]["max"]
                    min_value = pyro.param(f"{name}_{p}_min", torch.tensor(_min).float())
                    max_value = pyro.param(f"{name}_{p}_max", torch.tensor(_max).float()) 
                    v = pyro.sample(f"{name}_{p}", dist.Uniform(min_value, max_value))
                    params[name].update({p: v})
                elif p=="t": # EPL slope
                    _mean = prior_params["epl_slope"]["mean"]
                    _sigma = prior_params["epl_slope"]["sigma"]
                    mean = pyro.param(f"{name}_{p}_mu", torch.tensor(_mean).float())
                    sigma = pyro.param(f"{name}_{p}_sigma", torch.tensor(_sigma).float())
                    v = pyro.sample(f"{name}_{p}", dist.Normal(mean, sigma))
                    params[name].update({p: v})
                elif p=="b": # Einstein radius
                    _min = prior_params["einstein_radius"]["min"]
                    _max = prior_params["einstein_radius"]["max"]
                    min_value = pyro.param(f"{name}_{p}_min", torch.tensor(_min).float())
                    max_value = pyro.param(f"{name}_{p}_max", torch.tensor(_max).float())
                    v = pyro.sample(f"{name}_{p}", dist.Uniform(min_value, max_value))
                    params[name].update({p: v})
                elif p=="Ie": # Effective intensity
                    _min = prior_params[f"{name.lower()}_intensity"]["min"]
                    _max = prior_params[f"{name.lower()}_intensity"]["max"]
                    min_value = pyro.param(f"{name}_{p}_min", torch.tensor(_min).float())
                    max_value = pyro.param(f"{name}_{p}_max", torch.tensor(_max).float()) 
                    v = pyro.sample(f"{name}_{p}", dist.Uniform(min_value, max_value))
                    params[name].update({p: v})
                else:
                    raise ValueError(f"Parameter {name}_{p} does not have a prior")
                
    return params
