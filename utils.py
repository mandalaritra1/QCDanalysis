import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit
import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
import vector
print("awkward version ", ak.__version__)
print("coffea version ", coffea.__version__)
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from collections import defaultdict
import pickle
from distributed.diagnostics.plugin import UploadDirectory
import os
from plot_utils import adjust_plot
import matplotlib.pyplot as plt


def computeJER(pt, eta, rho, filename):
    df = pd.read_csv( filename, delimiter='\s+', skiprows = 1, names = ['eta_low','eta_high', 'rho_low', 'rho_high', 'unknown','pt_low','pt_high','par0','par1','par2','par3'])
    
    df = df[ (eta > df['eta_low']) &  (eta <= df['eta_high']) & (rho > df['rho_low']) & (rho <= df['rho_high'])  ]
    p0 = df['par0']
    p1 = df['par1']
    p2 = df['par2']
    p3 = df['par3']
    x = pt
    return np.sqrt(p0*np.abs(p0)/(x*x)+p1*p1*np.power(x,p3) + p2*p2)

class QCDProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        frac_axis = hist.axis.Regular(150, 0, 2.0, name="frac", label=r"Fraction")
        ptgen_axis = hist.axis.Variable([200,260,350,460,550,650,760,13000], name="ptgen", label=r"p_{T,RECO} (GeV)")
        n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        pt_axis = hist.axis.Variable([10,20,30,40,50,60,70,80,90,
                                     100,120,140,160,180,
                                     200,250,300,350,400,450,500,
                                     600,700,800,900,1000,
                                     1500,2000,3000], name="pt", label=r"$p_{T}$ [GeV]") #erased 4000 and 5000

        pileup_axis = hist.axis.Variable([0,10,20,30,40,50,60,90],name = "pileup", label = r"$\mu$" )
        #eta_axis = hist.axis.Regular(15, -4,4, name = "eta", label = r"$eta$")
        # eta_axis = hist.axis.Variable([0, 0.261, 0.522, 0.783,  1.044, 1.305, 1.566, 1.74, 1.93, 2.043, 2.172, 2.322, 2.5, 2.65, 2.853,
        #                               2.964, 3.139, 5],name = "eta", label = r"$\eta$")
        #eta_axis = hist.axis.Variable([-5.191, -3.839, -3.489, -3.139, -2.964, -2.853, -2.65, -2.5, -2.322,-2.172,-2.043, -1.93, -1.74, -1.566,-1.305,-1.044 ,-0.783 ,-0.522, -0.261, 0, 0.261, 0.522, 0.783, 1.044, 1.305, 1.566, 1.74, 1.93, 2.043, 2.172, 2.322, 2.5, 2.65, 2.853,], name = "eta", label = r"$\eta$")
        
        #eta_axis = hist.axis.Variable([0, 1.305,  2.5, 2.65, 2.853,
                                        #5.191],name = "eta", label = r"$\eta$")
        
        eta_axis = hist.axis.Variable([ 0, 0.5, 0.8, 1.1, 1.3, 1.7, 1.9, 2.1, 2.3, 2.5, 2.8, 3, 3.2, 4.7],name = "eta", label = r"$\eta$")
        
        
        rho_axis = hist.axis.Variable( [0, 7.47, 13.49, 19.52, 25.54, 31.57, 37.59, 90], 
                                      name = 'rho', label = r'$\rho$')
        
        jer_axis = hist.axis.Regular(100, 0.995, 1.030, name = 'jer', label = "JER" )
        
        
        h_njet_gen = hist.Hist(dataset_axis, n_axis, storage="weight", label="Counts")
        h_njet_reco = hist.Hist(dataset_axis, n_axis, storage="weight", label="Counts")
        
        h_pt_reco_over_gen = hist.Hist( dataset_axis, pt_axis, frac_axis, eta_axis, rho_axis, storage = "weight", label = "Counts")
        #h_pt_reco_over_raw = hist.Hist( dataset_axis, pt_raw_axis,n_axis, frac_axis, eta_axis, pileup_axis, storage = "weight", label = "Counts")
        
        response_pt = {}
        for i in range(len(pt_axis.centers)):
            response_pt[str(i)] = np.array([])
        
        
        #self.df = pd.read_csv( "Summer19UL17_JRV2_MC_PtResolution_AK4PFchs.txt", delimiter='\s+', skiprows = 1, names = ['eta_low','eta_high', 'rho_low', 'rho_high', 'unknown','pt_low','pt_high','par0','par1','par2','par3'])
    
        self.n_pt_bins = len(pt_axis.centers)
        self.pt_edges = pt_axis.edges
        
        cutflow = {}
        
        self.hists = {
            "njet_gen":h_njet_gen,
            "njet_reco":h_njet_reco,
            "pt_reco_over_gen": h_pt_reco_over_gen,
            "response_pt": response_pt,
            "cutflow":cutflow
        }
        
    @property
    def accumulator(self):
        return self.hists, self.n_pt_bins, self.pt_edges
    
    def process(self, events):
        dataset = events.metadata['dataset']
        
        if dataset not in self.hists["cutflow"]:
            self.hists["cutflow"][dataset] = defaultdict(int)
            


        gen_vtx = events.GenVtx.z
        reco_vtx = events.PV.z
        
        
        # delta_z < 0.2 between reco and gen
        events = events[np.abs(gen_vtx - reco_vtx) < 0.2]
        
        
        # loose jet ID
        events.Jet = events.Jet[events.Jet.jetId > 0]
        

        events = events[ak.num(events.Jet) > 0 ]
        dataset = events.metadata['dataset']
        
        genjets = events.GenJet[:,0:3]
        recojets = genjets.nearest(events.Jet, threshold = 0.2)
        
        sel = ~ak.is_none(recojets, axis = 1)
        
        genjets = genjets[sel]
        recojets = recojets[sel]
             
        ptresponse = recojets.pt/genjets.pt
        
        n_reco_vtx = events.PV.npvs #the number of primary vertices
        n_pileup = events.Pileup.nPU #number of pileupss
        rho = events.fixedGridRhoFastjetAll

        sel = ~ak.is_none(ptresponse,axis=1)
        ptresponse = ptresponse[sel]
        recojets = recojets[sel]
        genjets = genjets[sel]
        
        sel2 = ak.num(ptresponse) > 2
        
        recojets = recojets[sel2]
        genjets = genjets[sel2]
        
        ptresponse = ptresponse[sel2]
        ptresponse_raw = (recojets.pt * (1 - recojets.rawFactor))/genjets.pt
        
        n_reco_vtx = n_reco_vtx[sel2]
        n_pileup = n_pileup[sel2]
        rho = rho[sel2]
        
        n_reco_vtx = ak.broadcast_arrays(n_reco_vtx, recojets.pt)[0]
        n_pileup = ak.broadcast_arrays(n_pileup, recojets.pt)[0]
        rho = ak.broadcast_arrays(rho, recojets.pt)[0]
        
        
        ### finding median response here ##
        # for i in range(self.n_pt_bins):
        #     self.hists["response_pt"][str(i)] = np.append((self.hists["response_pt"][str(i)],ak.flatten(ptresponse)[np.digitize(ak.flatten(genjets.pt), self.pt_edges) == i ].to_numpy()),axis = None)
        # for i in range(self.n_pt_bins):
        #     with open(str(i)+".txt","a") as f:
        #         for item in ak.flatten(ptresponse)[np.digitize(ak.flatten(genjets.pt), self.pt_edges) == i ].to_numpy():
        #             f.write("%s\n" % item)
                
        
        self.hists["pt_reco_over_gen"].fill( dataset = dataset, pt = ak.flatten(genjets.pt),frac = ak.flatten(ptresponse), 
                                            rho = ak.flatten(rho), eta = np.abs(ak.flatten(genjets.eta)))
        
        #self.hists["pt_reco_over_raw"].fill( dataset = dataset, pt_raw = ak.flatten(recojets.pt*(1 - recojets.rawFactor)), n = ak.flatten(n_reco_vtx) ,frac = ak.flatten(ptresponse_raw), eta = np.abs(ak.flatten(genjets.eta)), pileup = ak.flatten(n_pileup))
        
            
        return self.hists
    
    def postprocess(self, accumulator):
        return accumulator
        
        
class Histfit:
    def __init__(self, hist_frac_pt, frac_values, pt_values):
        self.frac_values = frac_values
        self.hist_frac_pt = hist_frac_pt
        self.pt_values = pt_values
        
        self.parameters = {"mean":np.full(len(self.pt_values), None), "sigma": np.full(len(self.pt_values), None), "const":np.full(len(self.pt_values), None),"sigmaErr":np.full(len(self.pt_values), None)}
        
    def gauss(self,x,  x0, sigma,a):
        return (a*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)))
    
    def fitGauss(self, hist_frac, frac_values):
        parameters, covariance = curve_fit(self.gauss, frac_values, hist_frac) #,bounds = ([0.5,0.05,-5],[2,0.3,20])
        mean = parameters[0]
        sigma = parameters[1]
        const = parameters[2]
        meanErr = covariance[0][0]
        sigmaErr = covariance[1][1]
        return mean,sigma,const, sigmaErr
    
    def initiate_parameters(self):
        for i in range(len(self.hist_frac_pt)):
            hist_frac = self.hist_frac_pt[i]
            results = self.fitGauss(hist_frac, self.frac_values)
            for j,key in enumerate(self.parameters.keys()):
                self.parameters[key][i] = results[j]
                
    def store_parameters(self):
        self.initiate_parameters()
        for repeater in range(10):
            for i,hist_frac in enumerate(self.hist_frac_pt):
                sel = (self.frac_values > (self.parameters["mean"][i] - 1.5*self.parameters["sigma"][i])) &  (self.frac_values < (self.parameters["mean"][i] + 1.5*self.parameters["sigma"][i]))
                frac_values = self.frac_values[sel]
                hist_frac = hist_frac[sel]

                results = self.fitGauss(hist_frac, frac_values)
                if np.abs(results[2] - self.parameters["sigma"][i] ) < 0.000001:
                    break
                for j,key in enumerate(self.parameters.keys()):
                    self.parameters[key][i] = results[j]
    def show_fit(self, i):
        hist_frac = self.hist_frac_pt[i]
        sel = (self.frac_values > (self.parameters["mean"][i] - 1.5*self.parameters["sigma"][i])) &  (self.frac_values < (self.parameters["mean"][i] + 1.5*self.parameters["sigma"][i]))
        frac_values = self.frac_values[sel]
        hist_frac = hist_frac[sel]
        results = self.fitGauss(hist_frac, frac_values)
        
        print("Mean: {} ".format(results[0]))
        print("Width: {}".format(results[1]))
        for j,key in enumerate(self.parameters.keys()):
                    self.parameters[key][i] = results[j]
                
        plt.plot(self.frac_values, self.hist_frac_pt[i], 'b-', label = "Response")
        plt.plot(frac_values, self.gauss(frac_values, results[0], results[1], results[2]), 'black',linestyle = '--' ,label = "Gauss Fit")
            
        