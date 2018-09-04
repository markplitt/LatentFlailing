import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import utilities as u
import single_session_plots as ssp
import numpy as np
import scipy as sp
import sklearn as sk
from matplotlib import pyplot as plt


def make_spline_basis(x,knots=np.arange(0,450,100)):
    '''make cubic spline basis functions'''
    knotfunc = lambda k: np.power(np.multiply(x-k,(x-k)>0),3)
    spline_basis_list = [knotfunc(k) for k in knots.tolist()]
    spline_basis_list += [np.ones(x.shape[0]),x,np.power(x,2)]
    return np.array(spline_basis_list).T



def pos_morph_design_matrix(x,m,splines=True,knots=np.arange(-50,450,50)):
    '''make design matrix for GLM that uses basis functions for position and separate regresssors for each context'''
    if splines:
        basis = make_spline_basis(x,knots=knots)
    else:
        # add functionality for radial basis functions
        pass

    M = np.matlib.repmat(m[np.newaxis].T,1,basis.shape[1])

    dmat = np.hstack((basis,np.multiply(M,basis)))
    return dmat


def empirical_density(x,y,xbinsize=10):
    '''calculate empirical joint density'''
    if len(x.shape)>1:
        raise Exception('only dealing with univariate for now')

    # get ybins
    ymin = np.floor(y.ravel().min())
    ymax = np.ceil(y.ravel().max())
    ybins = np.linspace(ymin,ymax,20)
    y_binned = np.digitize(y,ybins,right=True)

    # get xbins
    xbins = [0]
    for i in range(xbinsize,int(np.ceil(x.ravel().max())+xbinsize),xbinsize):
        xbins.append(i)
    x_binned = np.digitize(x,xbins,right=True)

    #conditional distribution - P(Y|X)
    mu_y_x = np.zeros([len(xbins),])
    P_y_x= np.zeros([len(xbins),ybins.shape[0]])
    for b in np.unique(x_binned).tolist():
        inds = np.where(x_binned==b)[0]
        yx = y_binned[inds]
        mu_y_x[b] = y[inds].ravel().mean()
        y_inds = np.unique(yx)
        bcount = np.bincount(yx)
        bcount = bcount[bcount>0]

        P_y_x[b,y_inds] = bcount/x.shape[0]

    return P_y_x, mu_y_x


def gaussian_pdf(x,mu,sigma,univariate=True,eps=.01):
    '''calculate pdf for given mean and covariance'''
    if univariate:
        # add a spmall epsilon
        return 1/(2.*np.pi)**.5 * np.divide(np.exp(-1.*np.divide(np.power(x-mu,2),2*np.power(sigma,2))),sigma)
    else:
        # add a multivariate gaussian here

        # check for poor conditioning of covariance matrix
        pass

def transition_prob_matrix(x,binsize=5):
    '''calculate transition proabibilities'''
    # bin positions in 10 cm
    bin_edges = [0]
    for i in range(binsize,465,binsize):
        bin_edges.append(i)

    x_binned = np.digitize(x,bin_edges,right=True)

    # #transition matrix
    XX = np.zeros([len(bin_edges),len(bin_edges)])

    for b in np.unique(x_binned).tolist():
        inds = np.where(x_binned==b)[0]
        xx = x_binned[(inds+1)%x_binned.shape[0]]
        next_inds = np.unique(xx)
        bcount = np.bincount(xx)
        bcount = bcount[bcount>0]

        XX[next_inds,b] = bcount/bcount.sum()
    return XX, bin_edges

def forward_procedure_single_cell(XX_I0,XX_I1,L_I0,L_I1,II):
    '''calculate likelihood of data using forwad procedure. assuming equally
    likely to start in either context and probability that you start at the beginning
    of the track is 1'''

    alpha = np.zeros([XX_I0.shape[0],II.shape[0]])
    alpha[0,:] = .5


    for t in range(L_I0.shape[1]):
        la0 = np.multiply(L_I0[:,t],alpha[:,0])
        xla0 = np.dot(XX_I0,la0)

        la1 = np.multiply(L_I1[:,t],alpha[:,1])
        xla1 = np.dot(XX_I1,la1)


        A = np.zeros([II.shape[0],XX_I0.shape[0]])
        A[0,:] = xla0
        A[1,:] = xla1

        alpha = np.dot(II,A).T

    return alpha.ravel().sum()

def decoding_model(trial_C_z,XX_I0,XX_I1,mu_i0,mu_i1,alpha,beta,morphs):

    # allocate for single cell data
    post_i0x_y, post_i1x_y= [],[]
    post_i0 ,post_i1 = [],[]

    # allocate for population data
    pop_post_i0x_y, pop_post_i1x_y = [], []
    pop_post_i0, pop_post_i1 = [], []

    for trial,I  in enumerate(morphs):
        a,b  = alpha[trial,:], beta[trial,:]
        #print(a.shape)
        if trial%5==0:
            print("processing trial %d" % trial)


        cz = trial_C_z[trial]
        post_trial0,post_trial1 = [],[]
        pop_post_trial0, pop_post_trial1 = [],[]
        for j in range(cz.shape[0]):
            if j==0: # if first timepoint, set initial conditions

                #### single cell initial conditions
                # set probability of being in current position to 1
                onehot = .001*np.ones([XX_I0.shape[1],1])
                onehot[0] = 1.
                onehot = onehot/onehot.ravel().sum()

                # multiply by prior on being in context (.5)
                tmp0 = .5*np.dot(onehot,np.ones([1,cz.shape[1]]))
                tmp1 = .5*np.dot(onehot,np.ones([1,cz.shape[1]]))

                # normalization factor to account for digitization of position
                tmp_denom = tmp0.sum(axis=0)+tmp1.sum(axis=0)
                tmp_denom = np.dot(np.ones([XX_I0.shape[0],1]),tmp_denom[np.newaxis])

                # posterior having observed 0 time points
                Z0_t = np.divide(tmp0,tmp_denom)
                Z1_t = np.divide(tmp1,tmp_denom)


                #### pop decoding initial conditions
                ttmp0 = .5*onehot
                ttmp1 = .5*onehot

                # normalization factor
                ttmp_denom = ttmp0.sum(axis=0)+ttmp1.sum(axis=0)

                # posterior having observed 1 time frame
                ZZ0_t = ttmp0/ttmp_denom
                ZZ1_t = ttmp1/ttmp_denom

            ######## single cell decoding
            # alpha is trial x cell
            A = np.matlib.repmat(a[np.newaxis],XX_I0.shape[0],1)
            #print(A.shape,Z0_t.shape)
            B = np.matlib.repmat(b[np.newaxis],XX_I1.shape[0],1)
            XZ0 = np.dot(XX_I0,np.multiply(A,Z0_t)) + np.dot(XX_I1,np.multiply(1-A,Z1_t))
            XZ1 = np.dot(XX_I0,np.multiply(1-B,Z0_t)) + np.dot(XX_I1,np.multiply(B,Z1_t))

            # make activity into a matrix and means at each position into a matrix in order to calculate likelihoods
            CZX = np.matlib.repmat(cz[j,:],mu_i0.shape[0],1)

            l0 = gaussian_pdf(CZX,mu_i0,1)
            l1 = gaussian_pdf(CZX,mu_i1,1)
            denom = np.matlib.repmat(l0.sum(axis=0) + l1.sum(axis=0),mu_i0.shape[0],1)
            l0 = np.maximum(np.divide(l0,denom),.001)
            l1 = np.maximum(np.divide(l1,denom),.001)

            # numerator of new posterior
            tmpnum0 = np.multiply(XZ0,l0)
            tmpnum1= np.multiply(XZ1,l1)

            # normalization factor for updated posterior
            tmp_denom = tmpnum0.sum(axis=0)+tmpnum1.sum(axis=0)
            tmp_denom = np.dot(np.ones([XX_I0.shape[0],1]),tmp_denom[np.newaxis])

            # new posterior
            Z0_t = np.divide(tmpnum0,tmp_denom)
            Z1_t = np.divide(tmpnum1,tmp_denom)


            # add to list for trial
            post_trial0.append(Z0_t)
            post_trial1.append(Z1_t)


            # ######## population decoding
            XXZZ0 = np.dot(XX_I0,1*ZZ0_t) + np.dot(XX_I1,(1-1)*ZZ1_t)
            XXZZ1 = np.dot(XX_I0,(1-1)*ZZ0_t) + np.dot(XX_I1,1*ZZ1_t)

            # make activity into a matrix and means at each position into a matrix in order to calculate likelihoods
            CCZZXX = np.matlib.repmat(cz[j,:],mu_i0.shape[0],1)

            #calculate likelihoods as a function of binned position
            ll0 = gaussian_pdf(CCZZXX,mu_i0,1)
            ll1 = gaussian_pdf(CCZZXX,mu_i1,1)

            # normalize from binning
            ddenom = np.matlib.repmat(ll0.sum(axis=0) + ll1.sum(axis=0),mu_i0.shape[0],1)
            ll0 = np.divide(ll0,ddenom)
            ll1 = np.divide(ll1,ddenom)


            # population log-likelihood of current activity as a function of position
            log_L0 = np.log(ll0).sum(axis=1)
            log_L1 = np.log(ll1).sum(axis=1)

            # numerator of new posterior
            # first calculate in log space
            log_tmpnum0 = log_L0 + np.squeeze(np.log(XXZZ0))
            # bring back to values that won't overflow
            log_tmpnum0 -= log_tmpnum0.max()-1
            # back to normal space
            ttmpnum0 = np.exp(log_tmpnum0)

            # repeat
            log_tmpnum1 = log_L1 + np.squeeze(np.log(XXZZ1))
            log_tmpnum1 -= log_tmpnum1.max()-1
            ttmpnum1 = np.exp(log_tmpnum1)

            # normalization factor for updated posterior
            ttmp_denom = ttmpnum0.sum(axis=0)+ttmpnum1.sum(axis=0)


            # new posterior
            ZZ0_t = ttmpnum0/ttmp_denom
            ZZ1_t = ttmpnum1/ttmp_denom

            # add to list for trial
            pop_post_trial0.append(ZZ0_t)
            pop_post_trial1.append(ZZ1_t)




        # append trials posterior to list
        post_i0x_y.append(np.array(post_trial0))
        post_i1x_y.append(np.array(post_trial1))
        # sum across positions to get posterior of context
        post_i1.append(np.squeeze(np.array(post_trial1).sum(axis=1)))
        post_i0.append(np.squeeze(np.array(post_trial0).sum(axis=1)))


        # append trials population posterior to list
        pop_post_i0x_y.append(np.array(pop_post_trial0))
        pop_post_i1x_y.append(np.array(pop_post_trial1))
        # marginalize across position to get posterior of context
        pop_post_i1.append(np.squeeze(np.array(pop_post_trial1).sum(axis=1)))
        pop_post_i0.append(np.squeeze(np.array(pop_post_trial0).sum(axis=1)))


    return {'i0x_y':post_i0x_y,
            'i1x_y':post_i1x_y,
            'i0': post_i0,
            'i1':post_i1,
            'pop i0x_y': pop_post_i0x_y,
            'pop i1x_y':pop_post_i1x_y,
            'pop i0': pop_post_i0,
            'pop i1': pop_post_i1}
