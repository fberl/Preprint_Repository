
    for monkey in nonstrategic_monkeys:
        data = monkey_data[monkeys.index(monkey)]
        session_list = list(data['id'].unique())
        ls = 2*decay_order 
        ws = decay_order
        # cut off first 5 sessions
        data = data[data['id'] > session_list[5]]
        tcl = []
        tcw = []
        
        taus_ws = []
        taus_ls = []
        
        rsq_ws = []
        rsq_ls = []
        
        for session in list(data['id'].unique()):
            datum = data[data['id'] == session] 
            fit = fit_single_paper(datum,order=decay_order)
            ws_x = fit[ws:ls]
            ls_x = fit[ls:]
            # corr = np.corrcoef(ws_x,ls_x)[0,1]
            # corr = skewness(ls_x)
            # corr = ls_x[0]/max(ls_x)
            corr_ls = ls_x[0]/max(ls_x)
            corr_ws = ws_x[0]/max(ws_x)
            nonstrat_corr.append(corr_ls)
            tcl.append(corr_ls)
            tcw.append(corr_ws)
            
            # fit everything after first or second element to exponential decay. pick max between two as starting point
            ws_start_index = np.argmax(ws_x[:2])
            ls_start_index = np.argmax(ls_x[:2])
            t_ws = np.arange(0,len(ws_x) - ws_start_index,1)
            t_ls = np.arange(0,len(ls_x) - ls_start_index,1)
            exponential_decay_ws = curve_fit(exp_decay,t_ws,ws_x[ws_start_index:], p0 = [1,1,0])[0]
            exponential_decay_ls = curve_fit(exp_decay,t_ls,ls_x[ls_start_index:],p0 = [1,1,0])[0]
            tau_ws = exponential_decay_ws[1]
            tau_ls = exponential_decay_ls[1]
            
            taus_ws.append(tau_ws)
            taus_ls.append(tau_ls)
            
            wsrsq = 1 - sum((exp_decay(t_ws,*exponential_decay_ws) - ws_x[ws_start_index:])**2)/ sum((ws_x[ws_start_index:] - np.mean(ws_x[ws_start_index:]))**2)
            lsrsq = 1 - sum((exp_decay(t_ls,*exponential_decay_ls) - ls_x[ls_start_index:])**2)/ sum((ls_x[ls_start_index:] - np.mean(ls_x[ls_start_index:]))**2)
            
            rsq_ls.append(lsrsq)
            rsq_ws.append(wsrsq)
        # fit decay for coefficients across entire dataset
        # fit_all_sess = monkey_logistic_regression(df=data,order=decay_order,task='mp',monkey=monkey, plot=False)
        fit_all_sess = paper_logistic_regression(None,False,data=data,order=decay_order,err=False)
        ws_x = fit_all_sess[ws:ls]
        ls_x = fit_all_sess[ls:]
        ws_start_index = np.argmax(ws_x[:2])
        ls_start_index = np.argmax(ls_x[:2])
        t_ws = np.arange(0,len(ws_x) - ws_start_index,1)
        t_ls = np.arange(0,len(ls_x) - ls_start_index,1)
        exponential_decay_ws = curve_fit(exp_decay,t_ws,ws_x[ws_start_index:], maxfev=6000)[0]
        exponential_decay_ls = curve_fit(exp_decay,t_ls,ls_x[ls_start_index:],maxfev=6000)[0]
        tau_ws = exponential_decay_ws[1]
        tau_ls = exponential_decay_ls[1]
        wsrsq = 1 - sum((exp_decay(t_ws,*exponential_decay_ws) - ws_x[ws_start_index:])**2)/ sum((ws_x[ws_start_index:] - np.mean(ws_x[ws_start_index:]))**2)
        lsrsq = 1 - sum((exp_decay(t_ls,*exponential_decay_ls) - ls_x[ls_start_index:])**2)/ sum((ls_x[ls_start_index:] - np.mean(ls_x[ls_start_index:]))**2)

        # cosign similarity of ws and ls coefficients
        cossims[monkey] = np.dot(ws_x,ls_x)/(np.linalg.norm(ws_x)*np.linalg.norm(ls_x))
        
        
        
        
        corr_list.append({'monkey' : monkey, 'decay' :(tau_ws,tau_ls,wsrsq,lsrsq), 'non-monotonicity' : (np.mean(tcl),np.mean(tcw)), 'strategy': 'non-strategic'})
        # corr_list.append({'monkey' : monkey, 'decay' : [np.median(taus_ws),np.median(taus_ls),np.median(rsq_ws),np.median(rsq_ls)], 'non-monotonicity' : (np.mean(tcl),np.mean(tcw)), 'strategy': 'non-strategic'})
        # compute the correlation between WS and LS coefficients for each session
    for monkey in strategic_monkeys:
        data = monkey_data[monkeys.index(monkey)]
        session_list = list(data['id'].unique())

        ls = 2*decay_order 
        ws = decay_order
        # cut off first 5 sessions
        data = data[data['id'] > session_list[5]]
        tcw = []
        tcl = []
        
        taus_ws = []
        taus_ls = []
        
        rsq_ws = []
        rsq_ls = []
        for session in list(data['id'].unique()):
            datum = data[data['id'] == session] 
            fit = fit_single_paper(datum, order=decay_order)
            ws_x = fit[ws:ls]
            ls_x = fit[ls:]
            # corr = np.corrcoef(ws_x,ls_x)[0,1]
            # corr = skewness(ls_x)
            corr_ls = ls_x[0]/max(ls_x)
            corr_ws = ws_x[0]/max(ws_x)
            strat_corr.append(corr_ls)
            tcl.append(corr_ls)
            tcw.append(corr_ws)
            
            ws_start_index = np.argmax(ws_x[:2])
            ls_start_index = np.argmax(ls_x[:2])
            t_ws = np.arange(0,len(ws_x) - ws_start_index,1)
            t_ls = np.arange(0,len(ls_x) - ls_start_index,1)
            exponential_decay_ws = curve_fit(exp_decay,t_ws,ws_x[ws_start_index:], maxfev=6000)[0]
            exponential_decay_ls = curve_fit(exp_decay,t_ls,ls_x[ls_start_index:],maxfev=6000)[0]
            tau_ws = exponential_decay_ws[1]
            tau_ls = exponential_decay_ls[1]
            
            taus_ws.append(tau_ws)
            taus_ls.append(tau_ls)
            
            wsrsq = 1 - sum((exp_decay(t_ws,*exponential_decay_ws) - ws_x[ws_start_index:])**2)/ sum((ws_x[ws_start_index:] - np.mean(ws_x[ws_start_index:]))**2)
            lsrsq = 1 - sum((exp_decay(t_ls,*exponential_decay_ls) - ls_x[ls_start_index:])**2)/ sum((ls_x[ls_start_index:] - np.mean(ls_x[ls_start_index:]))**2)
            
            rsq_ls.append(lsrsq)
            rsq_ws.append(wsrsq)
            
            # corr_list.append({'monkey' : monkey, 'non-monotonicity' : corr, 'strategy': 'strategic'})
            #average across each monkey
        fit_all_sess = paper_logistic_regression(None,False,data=data,order=decay_order,err=False)
        ws_x = fit_all_sess[ws:ls]
        ls_x = fit_all_sess[ls:]
        ws_start_index = np.argmax(ws_x[:2])
        ls_start_index = np.argmax(ls_x[:2])
        t_ws = np.arange(0,len(ws_x) - ws_start_index,1)
        t_ls = np.arange(0,len(ls_x) - ls_start_index,1)
        exponential_decay_ws = curve_fit(exp_decay,t_ws,ws_x[ws_start_index:], maxfev=6000)[0]
        exponential_decay_ls = curve_fit(exp_decay,t_ls,ls_x[ls_start_index:],maxfev=6000)[0]
        tau_ws = exponential_decay_ws[1]
        tau_ls = exponential_decay_ls[1]
        wsrsq = 1 - sum((exp_decay(t_ws,*exponential_decay_ws) - ws_x[ws_start_index:])**2)/ sum((ws_x[ws_start_index:] - np.mean(ws_x[ws_start_index:]))**2)
        lsrsq = 1 - sum((exp_decay(t_ls,*exponential_decay_ls) - ls_x[ls_start_index:])**2)/ sum((ls_x[ls_start_index:] - np.mean(ls_x[ls_start_index:]))**2)
        cossims[monkey] = np.dot(ws_x,ls_x)/(np.linalg.norm(ws_x)*np.linalg.norm(ls_x))

        corr_list.append({'monkey' : monkey, 'decay' :(tau_ws,tau_ls,wsrsq,lsrsq), 'non-monotonicity' : (np.mean(tcl),np.mean(tcw)), 'strategy': 'strategic'})



stationarity_plot_ax = fig.add_subplot(gs[6:,8:])
all_ls = []
all_ws = []
for m in corr_list: # want each monkey to be a different color and be labeled by which monkey it is
    # then plot a scatter and a dashed line along diagonal
    if m['strategy'] == 'non-strategic':
        marker = 'o'
        color = 'tab:purple'
        lsns.append(np.mean(m['decay'][:2]))
        wsns.append(np.mean(m['decay'][2:]))
    else:
        marker = '*'
        color = 'tab:cyan'
        lss.append(np.mean(m['decay'][:2]))
        wss.append(np.mean(m['decay'][2:]))
    all_ls.append(np.mean(m['decay'][:2]))
    all_ws.append(np.mean(m['decay'][2:]))
    # lsm, wsm = m['non-monotonicity'] 
    stationarity_plot_ax.scatter(all_ls[-1],all_ws[-1], c = color, marker = marker, s=128)
    stationarity_plot_ax.text(all_ls[-1],all_ws[-1],m['monkey'], ha='right', va='bottom', fontsize = 12)
    
stationarity_plot_ax.set_xlabel(r'decay $\tau$', fontsize = 16)
stationarity_plot_ax.set_ylabel(r'decay $R^{2}$', fontsize = 16)
stationarity_plot_ax.set_title('Exponential Decay of Logistic Regression Coefficients', fontsize = 16)


complexity_ax = fig.add_subplot(gs[6:,6:8])
complexity_ax.set_title('Complexity of Monkey Behavior', fontsize = 16)
entropys = compute_entropy_and_mutual_information(monkey_data)
for m in entropys.keys():
    mutual_info = entropys[m]
    monkey = m
    if m == '112' or m == 112:
        monkey = 'F'
    if monkey in nonstrategic_monkeys:
        style = 'o'
        color = 'tab:purple'
    else:
        style = '*'
        color = 'tab:cyan'
    cossim = cossims[monkey]

    complexity_ax.scatter(mutual_info,cossim, label = monkey, c = color, marker = style)    
    complexity_ax.text(mutual_info,cossim,monkey, ha='right', va='bottom', fontsize = 12)

complexity_ax.set_xlabel('WS LS Sequence Mutual Information', fontsize = 16)
complexity_ax.set_ylabel('WS LS Similarity', fontsize = 16) # maybe should be similarity of regressors and not sequences


# https://stackoverflow.com/questions/67780602/annotate-underbracebrackets-between-points-on-the-x-axis

plt.tight_layout()
plt.show()
