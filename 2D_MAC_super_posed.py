import unittest
import numpy as np
from scipy.special import erfc

class CommSystemBase:
    """
    Base class for a communication system.
    It holds common parameters and methods to compute the Q-function, error rate, etc.
    """
    def __init__(self, P, sigma_n2, R, t, m):
        self.P = P              # Transmit power
        self.sigma_n2 = sigma_n2  # Noise variance
        self.R = R              # Data rate
        self.t = t              # Time or block length for tag
        self.m = m              # Time or block length for message

    def qfunc(self, x):
        """Q-function implemented using the complementary error function."""
        return 0.5 * erfc(x / np.sqrt(2))

    def error_rate(self, length, R, snr):
        """
        Compute the error rate using the provided formula.
        
        Parameters:
          length: the block length (t for tag or m for message)
          R: rate (could be adjusted depending on system)
          snr: the signal-to-noise ratio used in the computation.
        """
        p = self.qfunc(np.sqrt(2 * snr))
        # Note: In practice, one might add checks to avoid log2(0)
        C = 1 + p * np.log2(p) + (1 - p) * np.log2(1 - p)
        factor = np.sqrt(length / (p * (1 - p)))
        ratio = (C - R) / np.log2((1 - p) / p)
        return self.qfunc(factor * ratio)

    def snr(self):
        """Default SNR calculation (P/sigma_n2). Subclasses may override."""
        return self.P / self.sigma_n2

    def calculate_pe(self):
        """
        Calculate error probabilities (Pe) for tag and message.
        Should return a dictionary with keys "tag" and "msg", each containing energy and Pe.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def calculate_aer(self, Pem, Pet):
        """
        Calculate the average error rate (AER) given the message error probability (Pem)
        and tag error probability (Pet).
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def calculate_throughput(self, Pem, Pet):
        """
        Calculate the throughput (AT) given the message and tag error probabilities.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")





class TradCommSystem(CommSystemBase):
    """
    Traditional communication system.
    """
    def calculate_pe(self):
        gam = self.snr()
        # Tag: energy and error probability
        tag_energy = self.P / self.R
        tag_pe = self.error_rate(self.t, self.R, gam)
        # Message: energy and error probability
        msg_energy = self.P / self.R * (1 + self.t / self.m)
        msg_pe = self.error_rate(self.m, self.R, gam)
        return {"tag": {"energy": tag_energy, "pe": tag_pe},
                "msg": {"energy": msg_energy, "pe": msg_pe}}

    def calculate_aer(self, Pem, Pet):
        # AER for traditional: 1 - ((1-Pem)*(1-Pet))
        return 1 - ((1 - Pem) * (1 - Pet))

    def calculate_throughput(self, Pem, Pet):
        # Throughput for traditional: (m/(m+t)) * ((1-Pet)*(1-Pem))
        return (self.m / (self.m + self.t)) * ((1 - Pet) * (1 - Pem))


class OneDCommSystem(CommSystemBase):
    """
    1D communication system.
    """
    def __init__(self, P, sigma_n2, R, t, m, nr):
        super().__init__(P, sigma_n2, R, t, m)
        self.nr = nr  # Number of rows (spatial diversity parameter)

    def calculate_pe(self):
        gam = self.snr()
        tag_energy = self.P / self.R
        tag_pe = self.error_rate(self.t, self.R, gam)
        msg_energy = self.P / self.R * (1 + self.t / (self.nr * self.m))
        msg_pe = self.error_rate(self.m, self.R, gam)
        return {"tag": {"energy": tag_energy, "pe": tag_pe},
                "msg": {"energy": msg_energy, "pe": msg_pe}}

    def calculate_aer(self, Pem, Pet):
        # AER for 1D: 1 - (((1-Pem)**nr)*(1-Pet))
        return 1 - (((1 - Pem) ** self.nr) * (1 - Pet))

    def calculate_throughput(self, Pem, Pet):
        # Throughput for 1D: (nr*m/(nr*m+t)) * ((1-Pet)*((1-Pem)**nr))
        return (self.nr * self.m / (self.nr * self.m + self.t)) * ((1 - Pet) * ((1 - Pem) ** self.nr))


class TwoDCommSystem(CommSystemBase):
    """
    2D communication system.
    """
    def __init__(self, P, sigma_n2, R, t, m, nr, nc):
        super().__init__(P, sigma_n2, R, t, m)
        self.nr = nr  # Number of rows
        self.nc = nc  # Number of columns

    def calculate_pe(self):
        gam = self.snr()
        tag_energy = self.P / self.R
        tag_pe = self.error_rate(self.t, self.R, gam)
        msg_energy = self.P / self.R * (1 + (self.nr + self.nc) * self.t / (self.nr * self.nc * self.m))
        msg_pe = self.error_rate(self.m, self.R, gam)
        return {"tag": {"energy": tag_energy, "pe": tag_pe},
                "msg": {"energy": msg_energy, "pe": msg_pe}}

    @staticmethod
    def pr_2D(Pem, Pet, nr, nc):
        """
        Auxiliary function for 2D systems.
        """
        term1 = (1 - (1 - Pem) ** (nc - 1)) * (1 - Pet)
        term2 = (1 - (1 - Pem) ** (nr - 1)) * (1 - Pet)
        return term1 * term2

    def calculate_aer(self, Pem, Pet):
        # AER for 2D: pr_2D*(1-Pem) + Pem
        pr2d = self.pr_2D(Pem, Pet, self.nr, self.nc)
        return pr2d * (1 - Pem) + Pem

    def calculate_throughput(self, Pem, Pet):
        # Throughput for 2D: (nr*nc*m/(nr*nc*m + nc*t)) * ((1 - pr_2D)*(1-Pem))
        pr2d = self.pr_2D(Pem, Pet, self.nr, self.nc)
        return (self.nr * self.nc * self.m / (self.nr * self.nc * self.m + self.nc * self.t)) * ((1 - pr2d) * (1 - Pem))


class TwoDSuperCommSystem(CommSystemBase):
    """
    2D superimposed communication system.
    Note that this system uses a different SNR computation and energy formula.
    """
    def __init__(self, P, sigma_n2, R, t, m, nr, nc, alpha):
        super().__init__(P, sigma_n2, R, t, m)
        self.nr = nr      # Number of rows
        self.nc = nc      # Number of columns
        self.alpha = alpha  # Power splitting factor

    def snr_super_tag(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        """SNR for tag in the 2D superimposed system."""
        # Assuming that the message cancelation happends
        return self.alpha * self.P / self.sigma_n2
        # Without message cancelation
        #return self.alpha * self.P / (self.sigma_n2 + (1-self.alpha) * self.P)


    def snr_super_msg(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        """SNR for message in the 2D superimposed system."""
        return (1 - self.alpha) * self.P / (self.alpha * self.P + self.sigma_n2)

    def calculate_pe(self):
        # Tag calculations: using superimposed formulas
        gam_tag = self.snr()
        tag_energy = self.P / self.R
        tag_pe = self.error_rate(self.t, self.R, gam_tag)

        # Note: For tag error rate, the rate parameter is adjusted (e.g., R*t/(nc*m))
        gam_super_tag = self.snr_super_tag()
        super_tag_energy = (1 - self.alpha) * self.P / self.R
        super_tag_pe = self.error_rate(self.t, self.R * self.t / (self.nc * self.m), gam_super_tag)

        # Message calculations
        gam_msg = self.snr_super_msg()
        msg_energy = ((1 - self.alpha) * self.P) / self.R * (1 + self.t / (self.nr * self.m))
        msg_pe = self.error_rate(self.m, self.R, gam_msg)
        return {"tag": {"energy": tag_energy, "pe": tag_pe},
                "super_tag": {"energy": super_tag_energy, "pe": super_tag_pe},
                "msg": {"energy": msg_energy, "pe": msg_pe}}

    @staticmethod
    def pr_2D_super(Pem, Pet, nr, nc, Pet_reg):
        """
        Auxiliary function for 2D superimposed systems.
        """
        term1 = (1 - (1 - Pem) ** (nc - 1)) * (1 - Pet)
        term2 = (1 - (1 - Pem) ** (nr - 1)) * (1 - Pet_reg)
        return term1 * term2


    def calculate_aer(self, Pem, Pet, Pet_reg):
        # AER for 2D superimposed: pr_2D_super*(1-Pem) + Pem
        pr2d_super = self.pr_2D_super(Pem, Pet, self.nr, self.nc, Pet_reg)
        return pr2d_super * (1 - Pem) + Pem

    def calculate_throughput(self, Pem, Pet, Pet_reg):
        # Throughput for 2D superimposed: (nr*nc*m/(nr*nc*m+nc*t)) * ((1 - pr_2D_super)*(1-Pem))
        pr2d_super = self.pr_2D_super(Pem, Pet, self.nr, self.nc, Pet_reg)
        return (self.nr * self.nc * self.m / (self.nr * self.nc * self.m + self.nc * self.t)) * ((1 - pr2d_super) * (1 - Pem))



class TestCommSystems(unittest.TestCase):
    def setUp(self):
        # Common parameters
        self.P = 10.0
        self.sigma_n2 = 1.0
        self.R = 2.0
        self.t = 5.0
        self.m = 3.0
        self.nr = 2
        self.nc = 4
        self.alpha = 0.3
        self.Pet_reg = 0.1

    def check_range(self, value, lower=0.0, upper=1.0):
        """Helper function to check that a value is within [lower, upper]."""
        self.assertGreaterEqual(value, lower)
        self.assertLessEqual(value, upper)

    def test_pe_range_traditional(self):
        trad = TradCommSystem(self.P, self.sigma_n2, self.R, self.t, self.m)
        pe = trad.calculate_pe()
        self.check_range(pe["tag"]["pe"])
        self.check_range(pe["msg"]["pe"])

    def test_aer_and_throughput_range_traditional(self):
        trad = TradCommSystem(self.P, self.sigma_n2, self.R, self.t, self.m)
        pe = trad.calculate_pe()
        Pem = pe["msg"]["pe"]
        Pet = pe["tag"]["pe"]
        aer = trad.calculate_aer(Pem, Pet)
        at = trad.calculate_throughput(Pem, Pet)
        self.check_range(aer)
        self.check_range(at)

    def test_pe_range_1D(self):
        oned = OneDCommSystem(self.P, self.sigma_n2, self.R, self.t, self.m, self.nr)
        pe = oned.calculate_pe()
        self.check_range(pe["tag"]["pe"])
        self.check_range(pe["msg"]["pe"])

    def test_aer_and_throughput_range_1D(self):
        oned = OneDCommSystem(self.P, self.sigma_n2, self.R, self.t, self.m, self.nr)
        pe = oned.calculate_pe()
        Pem = pe["msg"]["pe"]
        Pet = pe["tag"]["pe"]
        aer = oned.calculate_aer(Pem, Pet)
        at = oned.calculate_throughput(Pem, Pet)
        self.check_range(aer)
        self.check_range(at)

    def test_pe_range_2D(self):
        twod = TwoDCommSystem(self.P, self.sigma_n2, self.R, self.t, self.m, self.nr, self.nc)
        pe = twod.calculate_pe()
        self.check_range(pe["tag"]["pe"])
        self.check_range(pe["msg"]["pe"])

    def test_aer_and_throughput_range_2D(self):
        twod = TwoDCommSystem(self.P, self.sigma_n2, self.R, self.t, self.m, self.nr, self.nc)
        pe = twod.calculate_pe()
        Pem = pe["msg"]["pe"]
        Pet = pe["tag"]["pe"]
        aer = twod.calculate_aer(Pem, Pet)
        at = twod.calculate_throughput(Pem, Pet)
        self.check_range(aer)
        self.check_range(at)

    def test_pe_range_2D_super(self):
        twod_super = TwoDSuperCommSystem(self.P, self.sigma_n2, self.R, self.t, self.m, self.nr, self.nc, self.alpha)
        pe = twod_super.calculate_pe()
        self.check_range(pe["tag"]["pe"])
        self.check_range(pe["msg"]["pe"])
        self.check_range(pe["super_tag"]["pe"])


    def test_aer_and_throughput_range_2D_super(self):
        twod_super = TwoDSuperCommSystem(self.P, self.sigma_n2, self.R, self.t, self.m, self.nr, self.nc, self.alpha)
        pe = twod_super.calculate_pe()
        Pem = pe["msg"]["pe"]
        Pet = pe["tag"]["pe"]
        PeSupertag = pe["super_tag"]["pe"]
        aer = twod_super.calculate_aer(Pem, Pet, PeSupertag)
        at = twod_super.calculate_throughput(Pem, Pet, PeSupertag)
        self.check_range(aer)
        self.check_range(at)

    def test_alpha_zero_pe_super_tag(self):
        # When alpha is 0, no power is allocated to the tag, so the tag's error probability must be 1.
        twod_super = TwoDSuperCommSystem(self.P, self.sigma_n2, self.R, self.t, self.m, self.nr, self.nc, 0.0)
        pe = twod_super.calculate_pe()
        self.assertAlmostEqual(pe["tag"]["pe"], 1.0, places=6)

    def test_alpha_one_pe_super_msg(self):
        # When alpha is 1, no power is allocated to the message, so the message's error probability must be 1.
        twod_super = TwoDSuperCommSystem(self.P, self.sigma_n2, self.R, self.t, self.m, self.nr, self.nc, 1.0)
        pe = twod_super.calculate_pe()
        self.assertAlmostEqual(pe["msg"]["pe"], 1.0, places=6)

# ---------------------------
# Run Unit Tests
# ---------------------------
if __name__ == '__main__':
    unittest.main()