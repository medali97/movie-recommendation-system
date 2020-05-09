def SVDpp(trainset, n_factors=20, n_epochs=20, init_mean=0, init_std_dev=.1, lr_all=.007, reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, lr_yj=None, reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, reg_yj=None, random_state=None, verbose=False):
    #trainset = trainset
    # (re) Initialise baselines
    bu = bi = None
    # user biases
    #SGD
    # user biases
    cdef np.ndarray[np.double_t] bu
    # item biases
    cdef np.ndarray[np.double_t] bi
    # user factors
    cdef np.ndarray[np.double_t, ndim=2] pu
    # item factors
    cdef np.ndarray[np.double_t, ndim=2] qi
    # item implicit factors
    cdef np.ndarray[np.double_t, ndim=2] yj

    cdef int u, i, j, f
    cdef double r, err, dot, puf, qif, sqrt_Iu, _
    cdef double global_mean = trainset.global_mean
    cdef np.ndarray[np.double_t] u_impl_fdb

    cdef double lr_bu = lr_bu
    cdef double lr_bi = lr_bi
    cdef double lr_pu = lr_pu
    cdef double lr_qi = lr_qi
    cdef double lr_yj = lr_yj

    cdef double reg_bu = reg_bu
    cdef double reg_bi = reg_bi
    cdef double reg_pu = reg_pu
    cdef double reg_qi = reg_qi
    cdef double reg_yj = reg_yj

    bu = np.zeros(trainset.n_users, np.double)
    bi = np.zeros(trainset.n_items, np.double)

    rng = get_rng(random_state)

    pu = rng.normal(init_mean, init_std_dev,(trainset.n_users, n_factors))
    qi = rng.normal(init_mean, init_std_dev,(trainset.n_items, n_factors))
    yj = rng.normal(init_mean, init_std_dev,
                    (trainset.n_items, n_factors))
    u_impl_fdb = np.zeros(n_factors, np.double)

    for current_epoch in range(n_epochs):
        if verbose:
            print(" processing epoch {}".format(current_epoch))
        for u, i, r in trainset.all_ratings():

            # items rated by u. This is COSTLY
            Iu = [j for (j, _) in trainset.ur[u]]
            sqrt_Iu = np.sqrt(len(Iu))

            # compute user implicit feedback
            u_impl_fdb = np.zeros(n_factors, np.double)
            for j in Iu:
                for f in range(n_factors):
                    u_impl_fdb[f] += yj[j, f] / sqrt_Iu

            # compute current error
            dot = 0  # <q_i, (p_u + sum_{j in Iu} y_j / sqrt{Iu}>
            for f in range(n_factors):
                dot += qi[i, f] * (pu[u, f] + u_impl_fdb[f])

            err = r - (global_mean + bu[u] + bi[i] + dot)

            # update biases
            bu[u] += lr_bu * (err - reg_bu * bu[u])
            bi[i] += lr_bi * (err - reg_bi * bi[i])

            # update factors
            for f in range(n_factors):
                puf = pu[u, f]
                qif = qi[i, f]
                pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                qi[i, f] += lr_qi * (err * (puf + u_impl_fdb[f]) -
                                     reg_qi * qif)
                for j in Iu:
                    yj[j, f] += lr_yj * (err * qif / sqrt_Iu -
                                         reg_yj * yj[j, f])

    bu = bu
    bi = bi
    pu = pu
    qi = qi
    yj = yj
    return bu, bi, pu, qi, yj