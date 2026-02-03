 
  
 #include <oxstd.h>
 #include <oxfloat.h>
 #include <oxprob.h>
 #import <maximize>
 #import <solvenle>
 #include <oxdraw.h>
 
 /* Global variables */ 
 static decl s_vy;
 static decl ygen; 
 static decl vx;
 static decl vx1;
 static decl vx2;
 static decl vx3;
 static decl vz;
 static decl vz1;
 static decl vz2;
 static decl vz3;
 static decl s_mX; 
 static decl s_mZ;
 static decl s_mXr;
 static decl a_t;
 

 //log-likelihood function of the Beta regression model with
//precision parameter modeling.


 floglikBeta(const vP, const adFunc, const avScore, const amHess)
 {
	   decl kk1     = columns(s_mX);
       decl kk2     = columns(s_mZ); 
       decl eta     = s_mX*vP[0:(kk1-1)];
       decl delta   = s_mZ*vP[kk1:((kk1+kk2)-1)];
  	   decl mu      = exp(eta) ./ (1.0+exp(eta));
	   decl phi     = exp(delta);
	   decl ystar   = log( s_vy ./ (1.0-s_vy) );
	   decl munew   = polygamma(mu .*phi, 0) - polygamma((1.0-mu).*phi, 0);
	   decl ynewst  = mu .*ystar+log(1-s_vy);
	   decl munewst = (munew .*mu)+ polygamma((1.0-mu).*phi, 0)-polygamma(phi, 0);
	   decl T       = diag( exp(eta) ./ (1.0+exp(eta)) .^2 );
	   decl H       = diag(phi);
	   decl m_phi   = diag(phi);

	       adFunc[0] =  double(sumc( loggamma(phi) - loggamma(mu.*phi)
                       - loggamma((1-mu).*phi) + ((mu .*phi)-1) .* log(s_vy)
	                   + ( ((1-mu).*phi)-1 ) .* log(1-s_vy) ));

	//Analytical derivatives of the log-likelihood function
   if(avScore)
   {

		(avScore[0])[0:(kk1-1)]           = s_mX'*m_phi*T*(ystar-munew);
		(avScore[0])[kk1:((kk1+kk2)-1)]   = s_mZ'*H*(ynewst-munewst);
   }

	    if( isnan(adFunc[0]) || isdotinf(adFunc[0]) )
 		return 0;
   else
 		return 1; // 1 indica sucesso
 }




 
/* Log-likelihood function to generate the residuals within the envelope */

 floglikgen(const vP, const adFunc, const avScore, const amHess)
 {	
   decl kk1       = columns(s_mX);
   decl kk2       = columns(s_mZ); 
   decl eta       = s_mX*vP[0:(kk1-1)];
   decl delta     = s_mZ*vP[kk1:((kk1+kk2)-1)];
   decl mu        = exp(eta) ./ (1.0+exp(eta)); 
   decl phi       = exp(delta);
   decl ystar     = log( ygen ./ (1.0-ygen) );
   decl munew     = polygamma(mu.*phi, 0) - polygamma((1.0-mu).*phi, 0);                
   decl munewst   = (munew .*mu)+ polygamma((1.0-mu).*phi, 0)-polygamma(phi, 0);
   decl ynewst    = mu .*ystar+log(1-ygen);
   decl T         = diag( exp(eta) ./ (1.0+exp(eta)) .^2 );                              
   decl H         = diag(phi);                                                           
   decl m_phi     = diag(phi);                                                           
                                                                                           
	   adFunc[0] = double(sumc( loggamma(phi) - loggamma(mu.*phi)                         
               - loggamma((1-mu).*phi) + ((mu .*phi)-1) .* log(ygen)                       
	       + ( ((1-mu).*phi)-1 ) .* log(1-ygen) ));                                        
                                                                                           
	//Analytical derivatives of the log-likelihood function                                  
   if(avScore)                                                                             
   {                                                                                       
                                                                                           
		(avScore[0])[0:(kk1-1)] = s_mX'*m_phi*T*(ystar-munew);                            
		(avScore[0])[kk1:((kk1+kk2)-1)]   = s_mZ'*H*(ynewst-munewst);                              
   }                                                                                       
                                                                                           
	    if( isnan(adFunc[0]) || isdotinf(adFunc[0]) )                                      
 		return 0;                                                                          
   else                                                                                    
 		return 1; // 1 indica sucesso                                                      
 }                                                                                         





 
//information matrix
f_Fisher(const vP)
{

	decl  kk1       = columns(s_mX);
    decl  kk2       = columns(s_mZ); 
    decl  etahat    = s_mX*vP[0:(kk1-1)];
    decl  deltahat  = s_mZ*vP[kk1:((kk1+kk2)-1)];
  	decl  muhat     = exp(etahat) ./ (1.0+exp(etahat));
	decl  phihat    = exp(deltahat);
	decl  psi1      = polygamma(muhat .* phihat, 1);
	decl  psi2      = polygamma((1.0-muhat).* phihat, 1);
	decl  T         = diag( exp(etahat) ./ (1.0+exp(etahat)) .^2 );
	decl  W         = diag( (phihat .^2) .*(psi1+psi2)) * (T .^2);
	decl  H         = diag(phihat);
	decl  m_c       = diag(phihat .* ((psi1 .* muhat)-(psi2 .*(1-muhat))));
	decl  C         = m_c*T*H;
	decl  D         = diag(psi1 .*(muhat.^2)+psi2 .*(1.0-muhat).^2-polygamma(phihat,1));
	decl  M         = D*(H .^2);
	decl  m_inf     = (W~C)|(C'~M);

	return m_inf;
	
}


 


/* Restricted likelihood log function (only with Beta_0) - for calculating
 R^2 based on the likelihood ratio between the model with only the intercept and the complete model*/
 
 floglikr(const vP, const adFunc, const avScore, const amHess)
 {	
   decl eta    = s_mXr*vP[0:0]; 
   decl mu     = exp(eta) ./ (1.0+exp(eta)); 
   decl phi    = vP[1]; 
   decl ystar  = log( s_vy ./ (1.0-s_vy) );
   decl munew  = polygamma(mu*phi, 0) - polygamma((1.0-mu)*phi, 0);
   decl T      = diag( exp(eta) ./ (1.0+exp(eta)) .^2 );

   adFunc[0] = double( sumc( loggamma(phi) - loggamma(mu*phi)
               - loggamma((1-mu)*phi) + (mu*phi-1) .* log(s_vy)
	           + ( (1-mu)*phi-1 ) .* log(1-s_vy) ));

   if(avScore)
   {  
      (avScore[0])[0] = phi*s_mXr'*T*(ystar-munew); 
	  (avScore[0])[1] = double(sumc( polygamma(phi, 0) - mu .* 
		                 polygamma(mu*phi, 0) - (1.0-mu) .* polygamma( (1.0-mu)*phi, 0) +
			             mu .* log(s_vy) + (1.0-mu) .* log(1.0-s_vy) )); 
   }	     	  
 
   if( isnan(adFunc[0]) || isdotinf(adFunc[0]) ) 
 	return 0; 
 
   else
        return 1; /* 1 indica sucesso */	
 }
 

 
 
 //function generating response Y								    
gerador_Y( const v_mu, const v_phi)							 
{															 
	decl ci;												 
	decl cn   = rows(v_mu);										 
	decl vecY = zeros(cn,1);									 
	for ( ci = 0; ci < cn; ci++)							 
     {														 
	  decl p   = (v_mu[ci]*v_phi[ci]);						 
	  decl q   = ((1-v_mu[ci])*v_phi[ci]);					 
	  vecY[ci] = ranbeta(1,1,p,q);							 
	 }														 
	 return vecY;											 
}															 
  															 
 															     

 
 decl fpout;
 decl fpout1;
 decl fpout2;
 decl fpout3;
 decl fpout4;
 decl fpout5;
 decl fpout6;
 decl fpout7;
 decl fpout8;
 decl fpout9;
 decl fpout10;
 decl fpout11;
 decl fpout12;
 decl fpout13;
 decl fpout14;
 decl fpout15;
 decl fpout16;
 decl fpout17;


 main()
 {
      decl cn, Sample, p, vbeta, p_phi,v_theta,dim;
	  decl mmle, stderrors, zstats, Lambda; 

	  decl v_phi,  v_delta, mphihat;
	  decl v_eta, v_mu, cs,  ms_vy, cfailure;

	  decl mu, phi, data, k, kk, kk1, kk2, ir1, n_Par, g_liber, vp1, dfunc1, ybar, yvar, betaols, ynew, dExecTime;
	  decl psi1, psi2, psi3, Ve, muhat, phihat, etahat, deltahat, m_phih, LV;
	  decl W, T, vc, D, C, g, Hstar, fisherinv, pseudoR2, pseudoR2star, dfuncr;
	  decl ir2, vp2, vp, dfuncsat, ir4, vp4,  H;   
	  decl cook_2; 
	  decl resstar_3, Vcomb;
	  decl Q, f, ystar, mustar, GL, GL1, GL2, e, XQXinv, M;
	  decl va, Vi, b, B, h, A,Vu;  
	  decl resstar, resstar_1, resstar_2, resstar_4, resstar_4gen, res1gen;
	  decl pseudoR2LR, pseudoR2LRc ;
	  
	  decl L1, L2, Obsinver, Lbeta, Lphi, B1, B2, E, DeltaP, F, d; 
	  decl uphi, LF_P, meval, mevec, C_maxP, I_maxP, CP_t;
	  decl LF_P_beta, LF_P_phi , CP_beta_t, CP_phi_t;
	 
	  decl l1t, l2t,l3t, Obst, Delta_t;
	    
	  decl sctotal, scbeta, scphi, sdtotal, sdbeta, sdphi;
	  decl C_maxPbeta, I_maxPbeta,C_maxPphi, I_maxPphi;
	  decl Sy, DeltaR, LF_R, LF_R_beta, LF_R_phi, C_maxR, I_maxR, C_maxRbeta, C_maxRphi, I_maxRbeta, I_maxRphi, CR_t, CR_beta_t, CR_phi_t;
	  decl srtotal, srbeta, srphi, sx, P, DeltaC, LF_C, LF_C_beta, LF_C_phi, C_maxC, I_maxC, C_maxCbeta, C_maxCphi, I_maxCbeta, I_maxCphi, CC_t, CC_beta_t, CC_phi_t;
	  
   //********************** PRESS e P2 *********************************************************************//     	  
      decl Ajuste, PRESS, PRESS1, PRESS4, PRESS5, PRESS2, PRESS3, Yaux, Yaux2, Yaux3, Yaux4, Yaux5, SST, SST2, SST3, SST4, SST5, P2, P2_c, P2_2, P2_2c, P4, P4_c, P5, P5_c;
      decl Yaux1, SST1, P2_1, P2_1c, PRESSi, PRESS1i, PRESS2i, PRESS3i, PRESS4i, PRESS5i, P3, P2_3, P2_3c;
	  decl Vcomb_1, Vcom_1gen, Vgamma_1, Vgamma_1gen, resstar_5, ystar_2, mustar_2, resstar_6;
	  decl P2_4, P2_5, P2_4c, P2_5c, Menvelope5, Menvelope6, Vcomb_1gen, resstar_5gen, resstar_6gen;
	  decl res5_r, res5_min, res5_mean, res5_max, res5_inf, res5_sup;
	  decl res6_r, res6_min, res6_mean, res6_max, res6_inf, res6_sup;
	  decl Menvelope7, res7_r, res7_min, res7_mean, res7_max, res7_inf, res7_sup, Res7qq;
	  decl Menvelope8,res8_r, res8_min, res8_mean, res8_max, res8_inf, res8_sup, Res8qq;	  
	  decl Res5qq, Res6qq, ystargen_2, mustargen_2;
	  decl resstar_O1, resstar_O2, resstar_O3, resstar_O4, resstar_O5, resstar_O6, P2p, P2p_c, P2c, P2c_c, P2v, P2v_c;
	  decl hC,HstarV, hV, HstarC, Vgammagen, Vcombgen_1, Vgammagen_1;


//*********************VARIANCE STANDARDS*****************************************************************//
decl HstarV1,hV1, resstar_8, resstar_7,dgen, Dgen, HstarVgen,hVgen,HstarV1gen, hV1gen,resstar_8gen,resstar_7gen;

	  
   //********************** E N V E L O P E *********************************************************************//        
 	                                                                                                                      
 	  decl Menvelope1, Menvelope2,Menvelope3, Menvelope4, fail, ir3, j, a, dfuncgen,  cfailure2, cs2;                                
 	  decl psi1gen, psi2gen, psi3gen, muhatgen, phihatgen, deltahatgen, etahatgen, m_phihgen, i;                          
 	  decl Vugen,  ystargen,  mustargen, Vcombgen, uphigen;                                                               
      decl res1_r, res1_min, res1_mean, res1_max, res2_r, res2_min, res2_mean, res2_max, res4_r, res4, res4_inf, res4_sup, res4_min, res4_mean, res4_max;                                 
      decl res3_r, res3_min, res3_mean, res3_max, res1_inf, res1_sup, res2_inf, res2_sup, res3_inf, res3_sup ;                                                                         
      decl Tgen, Wgen, tempinvgen, hgen;                                                                     
 	  decl resstar_1gen , 	resstar_2gen, resstar_3gen, uphiest;                                                          
 	  decl Res1qq,	Res2qq,Res3qq, Res4qq, fpout5, fpout7;                                                           
	  decl pseudoR2c, Cook, AIC, BIC;

	  decl munew,ynewst,munewst;

	  decl pseudoR2_Estar, pseudoR2c_Estar, pseudoR2_E,  pseudoR2c_E,  pseudoR2_E1,  pseudoR2c_E1;
	  decl I_maxCP, I_maxCPbeta, I_maxCPphi, CCP_t, CCP_beta_t, CCP_phi_t, I_maxCMP, I_maxCMPbeta, I_maxCMPphi, CCMP_t, CCMP_beta_t, CCMP_phi_t;


	   

	 /* Inicia o RelÃ³gio */				        
    dExecTime = timer();
    //turn off warnings
    oxwarning( 0 );
   	// selection of generator type âGeorge Marsagliaâ
 	ranseed("GM");
	ranseed({1965, 2001}); 

			   
		 fpout    = fopen("BetaHetero_logit.txt","w");
		 fpout1   = fopen("Residuos_123.txt","w");
		 fpout2   = fopen("Limites.txt","w");
//		 fpout3   = fopen("BetaHeteroErroPadrao_logit.txt","w");
//		 fpout4   = fopen("BetaHeteroEstimativaslogit.txt","w");

         fpout16  = fopen("Residuos_456.txt","w");
		 fpout17  = fopen("etahat.txt","w");

	  	 fpout5   = fopen("Influencia_Total_BetaHetero.txt","w"); 
	  	 fpout6   = fopen("Influencia_BetaHetero.txt","w");
//		 fpout7   = fopen("Mudanca_95_99_Hetero_Beta.txt","w");
//  	 fpout10  = fopen("Influencia_Total_BetaHetero_covariavel.txt","w");
//   	 fpout11  = fopen("Influencia_BetaHetero_covariavel.txt","w");	
//		 fpout12  = fopen("Influencia_Total_BetaHetero_covariavel_precisao.txt","w");
//	     fpout13  = fopen("Influencia_BetaHetero_covariavel_precisao.txt","w");
//		 fpout14  = fopen("Influencia_Total_BetaHetero_covariavel_mediaprecisao.txt","w");
//	     fpout15  = fopen("Influencia_BetaHetero_covariavel_mediaprecisao.txt","w");


		 
    println("\n\t\t VERSSAO OX: ", oxversion() );                                            
        println( "\n\t\t |------- MODEL SELECTION-------| ");                       
        println("\n\t\t |-------  BETA REGRESSION MODEL -------|" );                       


	    fprint(fpout,"\n\t\t VERSSAO OX: ", oxversion() );                                            
        fprint(fpout, "\n\t\t |------- MODEL SELECTION-------| ");                       
        fprint(fpout,"\n\t\t |-------  BETA REGRESSION MODEL -------|" );                           


		 		/************	REAL DATA	************/

	      data = loadmat("CCA_OX_CENTER.txt"); // USE THE DATA AVAILABLE TO RUN THE CODE 

		 decl Ind=data[][0];   decl CT=data[][1];
		 decl GLI=data[][2];   decl HDL=data[][3];
		 decl RCH=data[][4];   decl HBGLI=data[][5];
		 decl ID=data[][6];    decl GEN=data[][7];
		 decl ALT=data[][8];   decl PSO=data[][9];
		 decl IMC=data[][10];  decl CIN=data[][11];
		 decl QUA=data[][12]; //  decl s_vy=data[][13];
//		 decl aa=data[][14];   decl bb=data[][15];
//		 decl cc=data[][16];   decl dd=data[][17];


				 
			   s_mX=1~(IMC)~sqrt(PSO)~(HBGLI);		 		//
                           s_mZ=1~log(PSO)~exp(GEN)~log(ID);  	 

  
			   cn = rows(data);									
//             s_mZ=ones(cn,1);                                  
               s_vy =  data[][13];
			   
		  
//		decl estimativ = loadmat("Beta_HeteroEstimativas.txt");
//		decl ErroPadrao = loadmat("Beta_HeteroErroPadrao.txt");
		kk = columns(data); 


		fprint(fpout,"s_vy",maxc(s_vy)~minc(s_vy));
		fprint(fpout,"mediana s_vy",quantilec(sortc(s_vy), <0.50>));

		Ajuste   = zeros(2,1);                           
		kk1      = columns(s_mX);	// number of parameters in the mena model
		kk2      = columns(s_mZ);  // number of parameters in the precision model   
		n_Par    = kk1+kk2;                                               
		g_liber  = cn-n_Par;                                           

		//enriched data matrix
		decl m_XAumentada = zeros(2*cn,n_Par);
		m_XAumentada[0:((cn)-1)][0:(kk1-1)] = s_mX;
		m_XAumentada[(cn):(2*cn-1)][kk1:(n_Par-1)] =s_mZ ;

    	        decl m_Xinv = invertsym(s_mX'*s_mX)*s_mX';   //auxiliary inversion in X
	        decl m_Zinv = invertsym(s_mZ'*s_mZ)*s_mZ';  //auxiliary inversion in Z
		ynew        = log( s_vy ./ (1.0-s_vy) );    //Y vector transformed for auxiliary regression INITIAL KICK

  //*****************************//
 
   	//auxiliary regression to determine initial points at the moment
   //of maximization by BFGS
		decl v_z         = log(s_vy ./ (1.0 - s_vy)); //transformed response v_z
		decl v_betaini   = m_Xinv*v_z;            //initial estimation of betas
		decl v_zEstim    = s_mX*v_betaini; 	  // estimated v_z vector
		decl v_eEstim    = v_z-v_zEstim;		 // estimated residual vector
		decl b_SomaQua   = v_eEstim'*v_eEstim;           // sum of squares
        decl v_muVee     = exp(v_zEstim) ./ (1.0+exp(v_zEstim));  //transformed mu vector
		decl v_sigma2Est = b_SomaQua/(g_liber);				  //*v_invgDerPrima2;
	    decl p_aux       = ((1.0 ./ (v_sigma2Est))*(v_muVee .* (1.0-v_muVee))); //initial phi
		decl v_gama      = m_Zinv*(log(p_aux));
		decl v_Ini       = v_betaini|v_gama; //initial theta vector for maximization
  
 //************* Maximum likelihood estimation *****************//
	  	 		 
		// initial values
	 		vp1 = v_Ini;
			//println(v_Ini);
		
		//  BFGS maximization parameters
        MaxControl( 50, -1 );
    
      ir1 = MaxBFGS(floglikBeta, &vp1, &dfunc1, 0, 0);
      println("\nCONVERGENCE STATUS: ", MaxConvergenceMsg(ir1));
//	  println(vp1);
	  
	  if(ir1 == MAX_CONV || ir1 == MAX_WEAK_CONV)
	  {
 		AIC       = -2*dfunc1+2*(n_Par);
        BIC       = -2*dfunc1+(n_Par)*log(cn);
	   	etahat    = s_mX*vp1[0:(kk1-1)];              
        deltahat  = s_mZ*vp1[kk1:((kk1+kk2)-1)];    
		muhat     = exp(etahat) ./ (1.0+exp(etahat)); 
	  	phihat    = exp(deltahat);


		 
		m_phih       = diag(phihat);
		H            = diag(phihat);
		psi1         = polygamma(muhat.*phihat, 1); 
	    psi2         = polygamma((1.0-muhat).*phihat, 1); 
		psi3         = polygamma(phihat, 1); 
		decl m_Q     = f_Fisher(vp1);
	    decl K_theta = m_XAumentada'*m_Q*m_XAumentada; //informacao de Fisher



		//matriz de variancia assintotica
		decl m_cov_Assint       = (K_theta)^(-1);
		decl m_cov_Assint_beta  = m_cov_Assint[0:(kk1-1)][0:(kk1-1)];
		decl m_cov_Assint_phi   = m_cov_Assint[kk1:(kk1+kk2)-1][kk1:(kk1+kk2)-1];

		T   = diag( exp(etahat) ./ (1.0+exp(etahat)) .^2 );
        W   = diag(phihat.*(psi1+psi2)).* (T.^2); 
        vc  = (phihat).*(psi1.*muhat-psi2.*(1.0-muhat));
		C   = diag(vc);
		d   = (psi1.*(muhat.^2)+psi2.*(1.0-muhat).^2-polygamma(phihat,1)).*(phihat.^2);
		D   = diag(d);

		
		decl tempinv      = invertsym(s_mZ'*D*s_mZ);
		decl Assint_beta  = invertsym(s_mX'*m_phih*W*s_mX);
	    decl K1           = invertsym(s_mX'*m_phih*W*s_mX - s_mX'*C*T*H*s_mZ*tempinv*s_mZ'*H'*T'*C'*s_mX);
		decl K2           = -K1*s_mX'*C*T*H*s_mZ*tempinv;
		decl K3           = tempinv*(unit(kk2)+ s_mZ'*H'*T'*C'*s_mX*K1*s_mX'*C*T*H*s_mZ*tempinv);
		fisherinv         = ((K1~K2)|(K2'~K3));
		decl stderrors    = sqrt(diagonal(fisherinv))'; 
		decl zstats       = vp1 ./ stderrors;




		ystar             = log( s_vy ./ (1.0-s_vy) );
	    munew             = polygamma(muhat .*phihat, 0) - polygamma((1.0-muhat).*phihat, 0);
	    ynewst            = muhat .*ystar+log(1-s_vy);
	    munewst           = (munew .*muhat)+ polygamma((1.0-muhat).*phihat, 0)-polygamma(phihat, 0);

	
		Lambda =  maxc(phihat) /minc(phihat);
		

	    fprint(fpout,"\n  ", "%10.4f", "%c", {"Estimativas", "Erros padroes", "IC inferior", "IC superior","P_valor"}, 
	    vp1~stderrors~(vp1-1.96*stderrors)~(vp1+1.96*stderrors)~2.0*(1.0-probn(fabs(zstats))));

		println("\n  ", "%12.4f",  "%r", {"Estimativas", "Erros padroes", "IC inferior", "IC superior","P_valor"}, 
	    vp1'|stderrors'|(vp1-1.96*stderrors)'|(vp1+1.96*stderrors)'|2.0*(1.0-probn(fabs(zstats)))');
                    


	    Hstar    = sqrt(W*m_phih)*s_mX*Assint_beta*s_mX'*sqrt(W*m_phih);
		ystar    = log( s_vy ./ (1.0-s_vy) ); 
	    mustar   = polygamma(muhat.*phihat, 0) - polygamma((1.0-muhat).*phihat, 0);
		d        = (psi1.*(muhat.^2)+psi2.*(1.0-muhat).^2-polygamma(phihat,1)).*(phihat.^2);
		va       = muhat.*(ystar-mustar)+ log(1.0-s_vy) - polygamma((1.0-muhat).*phihat, 0)	+ polygamma(phihat, 0);
		Vi       = diag(d - va.*phihat);
		Q        = diag((phihat.*( polygamma(muhat.*phihat, 1) + polygamma((1-muhat).*phihat, 1) ) - (ystar-mustar).*((1.0-2*muhat) ./ (muhat.*(1.0-muhat)) )).*(muhat.^2).*(1.0-muhat).^2
	    ); 
		vc       = (phihat).*(psi1.*muhat-psi2.*(1.0-muhat));
		f        = vc -(ystar-mustar);
		b        = (-(s_vy-muhat) ./ (s_vy .* (1-s_vy)));
		B        = diag(b);
		XQXinv   = invertsym(s_mX'*m_phih*Q*s_mX);
		M        = diag( 1 ./ (s_vy .* (1.0-s_vy)) );  
		F        = diag(f); 
		h        = diagonal(Hstar)'; 


/************************DEFINING THE PROJECTION MATRIX FOR Z */

		HstarV = s_mZ*invertsym(s_mZ'*s_mZ)*s_mZ';
		hV = diagonal(HstarV)';

		HstarV1 = sqrt(D)*s_mZ*invertsym(s_mZ'*D*s_mZ)*s_mZ'*sqrt(D);
		hV1 = diagonal(HstarV1)';

		
		
/****************************SELECTION CRITERIA R2_FC*/
			  
 			 /* pseudo-R2 */                                                  
			   pseudoR2  = (correlation(ynew~etahat)[0][1])^2;                 
			   pseudoR2c = 1.0 - (1 - pseudoR2)*((cn - 1)/g_liber);           
			                                                                  
//*************************************************// R2 based on log-likelihood  //***************************************************************//


			   s_mXr        = ones(cn, 1);                                           
			   decl ynewbar = meanc(ynew);                                    
			   decl muhatr  = exp(ynewbar)/(1.0 + exp(ynewbar));               
			   vp           = ynewbar|((1.0/(varc(ynew)*muhatr*(1.0-muhatr))));        


 	  		   ir2 = MaxBFGS(floglikr, &vp, &dfuncr, 0, 0);                                                           
 			                                                                                                           

			   if(ir2 == MAX_CONV || ir2==MAX_WEAK_CONV)                                                                    
 	  		   {                                                                                                       
 			       pseudoR2LR  = 1.0 - (exp(dfuncr)/exp(dfunc1))^(2/cn);                                                
      		   	   pseudoR2LRc = 1.0 -(1 - pseudoR2LR)*((cn - 1)/(cn - (1 + 0.4)*(kk1 + 1) - (1 - 0.4)*(kk2 + 1)));    
 	  		   }

			   
//*************************************************// Residuals  //***************************************************************//                        

			
			 Vu         = (psi1+psi2);
		         Vcomb      = (muhat.^2).*psi2	+ psi1.*(1.0 + muhat).^2 - psi3;
			 Vcomb_1 =  psi1  - psi3;
			 Vgamma_1 =  psi2	- psi3;
			 ystar_2 = log(1.0-s_vy) ;
			 mustar_2 = polygamma((1.0-muhat).*phihat, 0)	-  polygamma(phihat, 0);


			 resstar_1 = (ystar - mustar)./ sqrt(Vu); // Weighted Residual            
			 resstar_2 = resstar_1./sqrt(1.0-h);  // Standardized Weighted Residual
			 resstar_3 = ((ystar - mustar) +  va)./sqrt(Vcomb); // Combinated Residual
   			 resstar_5 = va ./ sqrt(psi1.*(muhat.^2)+psi2.*(1.0-muhat).^2-polygamma(phihat,1));	// Variance Residual
			 resstar_8 = va ./ sqrt((psi1.*(muhat.^2)+psi2.*(1.0-muhat).^2-polygamma(phihat,1)).*(1-hV1)); // Standardized Variance Residual
                    

			 
			 resstar_4 = ((ystar - mustar) +  (ystar_2-mustar_2))./sqrt(Vcomb_1);	 // Bias-Variance Residual
			 resstar_6 = (ystar_2-mustar_2)./sqrt(Vgamma_1);	 // EF-Variance Residual
			 resstar_7 = (ystar - mustar)./ sqrt(Vu); // Bias Residual			 

			

			
			
//********************************************************** Calculation of quantities for constructing envelope bands **********************************************************//                                                                  
			  			                                                                                                                                                                                 
		     Menvelope1 = zeros(cn,80); 
			 Menvelope2 = zeros(cn,80); 
			 Menvelope3 = zeros(cn,80);
			 Menvelope4 = zeros(cn,80);
			 Menvelope5 = zeros(cn,80);
			 Menvelope6 = zeros(cn,80);
			 Menvelope7 = zeros(cn,80);
			 Menvelope8 = zeros(cn,80);			 
				 

					  
				  	  fail = 0; 
				         for(j=0; j<80; j++)
				   	  {
				   	      ygen = zeros(cn, 1); 
				   	      for(i=0; i<cn; i++)
				   	      {
				                 ygen[i] = ranbeta(1, 1, muhat[i]*phihat[i], (1-muhat[i])*phihat[i]);  
				   		  }
				   		     			
				  	 
				  		vp4 = vp1; 
				  		
				   	   
				   	  ir4 = MaxBFGS(floglikgen, &vp4, &dfuncsat, 0, TRUE);
				   
				      if(ir4 == MAX_CONV || ir4 == MAX_WEAK_CONV)

							 {
							 
							
							 etahatgen = s_mX*vp4[0:(kk1-1)]; 
				             muhatgen = exp(etahatgen) ./ (1+exp(etahatgen));
				  			 decl deltahatgen   = s_mZ*vp4[kk1:((kk1+kk2)-1)]; 
				   		     phihatgen =exp(deltahatgen);			 
				   		     psi1gen = polygamma(muhatgen.*phihatgen, 1); 
				   		     psi2gen = polygamma((1.0-muhatgen).*phihatgen, 1); 
							 psi3gen = polygamma(phihatgen, 1);   
							 Vugen = (psi1gen+psi2gen);
				  			 Tgen = diag( exp(etahatgen) ./ (1.0+exp(etahatgen)) .^2 );
				             Wgen =diag( phihatgen.*(Vugen))*(Tgen .^2);
				  			 decl m_phihgen = diag(phihatgen);

							dgen   = (psi1gen.*(muhatgen.^2)+psi2gen.*(1.0-muhatgen).^2-polygamma(phihatgen,1)).*(phihatgen.^2);
							Dgen   = diag(dgen);

							HstarVgen = s_mZ*invertsym(s_mZ'*s_mZ)*s_mZ';
							hVgen = diagonal(HstarVgen)';		

							HstarV1gen = sqrt(Dgen)*s_mZ*invertsym(s_mZ'*Dgen*s_mZ)*s_mZ'*sqrt(Dgen);
							hV1gen = diagonal(HstarV1gen)';		
							 
							 
				  			 tempinvgen = invertsym(s_mX'*m_phihgen *Wgen*s_mX); 
				  			 hgen = diagonal(sqrt(Wgen*m_phihgen)*s_mX*tempinvgen*s_mX'*sqrt(Wgen*m_phihgen ))';
						   	 ystargen =  log( ygen ./ (1.0-ygen) );
				  			 mustargen = polygamma(muhatgen.*phihatgen, 0) - polygamma((1.0-muhatgen).*phihatgen, 0);
							 ystargen_2 =  log(1.0-ygen);                                                   
                             mustargen_2 = polygamma((1.0-muhatgen).*phihatgen, 0) - polygamma(phihatgen, 0); 
							 Vcombgen =  (muhatgen.^2).*psi2gen	+ psi1gen.*(1.0 + muhatgen).^2 - psi3gen;
				  			 uphigen = muhatgen.*(ystargen-mustargen)+ log(1.0-ygen)-polygamma((1.0-muhatgen).*phihatgen, 0)+ polygamma(phihatgen, 0);
							 Vgammagen =  (muhatgen.^2).*psi1gen + psi2gen.*((1.0 + muhatgen).^2)- psi3gen;
							 Vcombgen_1 =  psi1gen - psi3gen;
							 Vgammagen_1 =  psi2gen - psi3gen;

							 resstar_1gen =(ystargen - mustargen)./ sqrt(Vugen); 
							 resstar_2gen = resstar_1gen./sqrt(1.0-hgen);		 
							 resstar_3gen =((ystargen - mustargen) +  uphigen)./sqrt(Vcombgen);	
							 resstar_4gen = ((ystargen - mustargen) +  (ystargen_2 - mustargen_2))./sqrt(Vcombgen_1);      
							 resstar_5gen =	uphigen ./ sqrt(psi1gen.*(muhatgen.^2)+psi2gen.*(1.0-muhatgen).^2-polygamma(phihatgen,1)); 
 						     resstar_6gen =(ystargen_2 - mustargen_2)./sqrt(Vgammagen_1); 
							 resstar_7gen = (ystargen - mustargen)./ sqrt(Vugen);  						 
							 resstar_8gen =	uphigen ./ sqrt((psi1gen.*(muhatgen.^2)+psi2gen.*(1.0-muhatgen).^2-polygamma(phihatgen,1)).*(1-hV1gen)); 




							 
							  Menvelope1[][j] = resstar_1gen;  
						 	  Menvelope2[][j] = resstar_2gen;
							  Menvelope3[][j] = resstar_3gen;
							  Menvelope4[][j] = resstar_4gen;
							  Menvelope5[][j] = resstar_5gen;
							  Menvelope6[][j] = resstar_6gen;
							  Menvelope7[][j] = resstar_7gen; 
							  Menvelope8[][j] = resstar_8gen; 
				

							 }
				  	          	 
					   	      else 
					   	       {
					   	          ++fail;
					   	          --j;
					   	       } 

					      }	


                           Menvelope1 = sortc((Menvelope1)); // Sorting the residues generated by simulation          
						   res1_r     =   resstar_1;   // True Residual - No sorting 
                           res1_min   =   quantilec((Menvelope1'), <0.05>)'; // Lower band of the envelope           
						   res1_mean  =   quantilec((Menvelope1'), <0.50>)';  // Median                             
                           res1_max   =   quantilec((Menvelope1'), <0.95>)';  // Upper band of the envelope          
                           res1_inf   =   quantilec((meanc(Menvelope1'))', <0.025>)';  //                            
                           res1_sup   =   quantilec((meanc(Menvelope1'))', <0.975>)';  //                          
						                                                                                             
						   Menvelope2 = sortc((Menvelope2));           
						   res2_r     =   resstar_2;    
						   res2_min   =   quantilec((Menvelope2'), <0.05>)';           
						   res2_mean  =   quantilec((Menvelope2'), <0.50>)';           
						   res2_max   =   quantilec((Menvelope2'), <0.95>)';           
						   res2_inf   =   quantilec((meanc(Menvelope2'))', <0.025>)';                         
						   res2_sup   =   quantilec((meanc(Menvelope2'))', <0.975>)';                              
						                                                                                             
						                                                                       
						                                                                                             
						   Menvelope3 = sortc((Menvelope3));       
						   res3_r     =   resstar_3;                
						   res3_min   =   quantilec((Menvelope3'), <0.05>)';            
						   res3_mean  =   quantilec((Menvelope3'), <0.50>)';            
						   res3_max   =   quantilec((Menvelope3'), <0.95>)';            
						   res3_inf   =   quantilec((meanc(Menvelope3'))', <0.025>)';  //                            
						   res3_sup   =   quantilec((meanc(Menvelope3'))', <0.975>)';  //

						   Menvelope4 = sortc((Menvelope4));        
						   res4_r     =   resstar_4;           
						   res4_min   =   quantilec((Menvelope4'), <0.05>)';           
						   res4_mean  =   quantilec((Menvelope4'), <0.50>)';           
						   res4_max   =   quantilec((Menvelope4'), <0.95>)';           
						   res4_inf   =   quantilec((meanc(Menvelope4'))', <0.025>)';  //                            
						   res4_sup   =   quantilec((meanc(Menvelope4'))', <0.975>)';  //

						   Menvelope5 = sortc((Menvelope5));     
						   res5_r     =   resstar_5;                   
						   res5_min   =   quantilec((Menvelope5'), <0.05>)';           
						   res5_mean  =   quantilec((Menvelope5'), <0.50>)';           
						   res5_max   =   quantilec((Menvelope5'), <0.95>)';           
						   res5_inf   =   quantilec((meanc(Menvelope5'))', <0.025>)';  //                            
						   res5_sup   =   quantilec((meanc(Menvelope5'))', <0.975>)';  //


						   Menvelope6 = sortc((Menvelope6));         
						   res6_r     =   resstar_6;                 
						   res6_min   =   quantilec((Menvelope6'), <0.05>)';          
						   res6_mean  =   quantilec((Menvelope6'), <0.50>)';          
						   res6_max   =   quantilec((Menvelope6'), <0.95>)';          
						   res6_inf   =   quantilec((meanc(Menvelope6'))', <0.025>)';  //                            
						   res6_sup   =   quantilec((meanc(Menvelope6'))', <0.975>)';  //
						   

						   Menvelope7 = sortc((Menvelope7));         
						   res7_r     =   resstar_7;                
						   res7_min   =   quantilec((Menvelope7'), <0.05>)';         
						   res7_mean  =   quantilec((Menvelope7'), <0.50>)';         
						   res7_max   =   quantilec((Menvelope7'), <0.95>)';         
						   res7_inf   =   quantilec((meanc(Menvelope7'))', <0.025>)';  //                            
						   res7_sup   =   quantilec((meanc(Menvelope7'))', <0.975>)';  //

						   Menvelope8 = sortc((Menvelope8));          
						   res8_r     =   resstar_8;   
						   res8_min   =   quantilec((Menvelope8'), <0.05>)';         
						   res8_mean  =   quantilec((Menvelope8'), <0.50>)';         
						   res8_max   =   quantilec((Menvelope8'), <0.95>)';         
						   res8_inf   =   quantilec((meanc(Menvelope8'))', <0.025>)';  //                            
						   res8_sup   =   quantilec((meanc(Menvelope8'))', <0.975>)';  //


						   
  }	
  
				   Res1qq = zeros(cn, 4);     
				   Res1qq[][3] = res1_r   ;        
				   Res1qq[][0] = res1_min ;        
				   Res1qq[][1] = res1_mean;        
				   Res1qq[][2] = res1_max ;        
				   Res2qq = zeros(cn, 4);     
				   Res2qq[][3] = res2_r   ;        
				   Res2qq[][0] = res2_min ;        
				   Res2qq[][1] = res2_mean;        
				   Res2qq[][2] = res2_max ;        
				   Res3qq = zeros(cn, 4);     
				   Res3qq[][3] = res3_r   ;        
				   Res3qq[][0] = res3_min ;        
				   Res3qq[][1] = res3_mean;        
				   Res3qq[][2] = res3_max ;        
				   Res4qq = zeros(cn, 4);     
				   Res4qq[][3] = res4_r   ;        
				   Res4qq[][0] = res4_min ;        
				   Res4qq[][1] = res4_mean;        
				   Res4qq[][2] = res4_max ;
				   Res5qq = zeros(cn, 4);     
				   Res5qq[][3] = res5_r   ;        
				   Res5qq[][0] = res5_min ;        
				   Res5qq[][1] = res5_mean;        
				   Res5qq[][2] = res5_max ;
				   Res6qq = zeros(cn, 4);     
				   Res6qq[][3] = res6_r   ;        
				   Res6qq[][0] = res6_min ;        
				   Res6qq[][1] = res6_mean;        
				   Res6qq[][2] = res6_max ;

				   Res7qq = zeros(cn, 4);     
				   Res7qq[][3] = res7_r   ;    
				   Res7qq[][0] = res7_min ;        
				   Res7qq[][1] = res7_mean;        
				   Res7qq[][2] = res7_max ; 

				   Res8qq = zeros(cn, 4);     
				   Res8qq[][3] = res8_r   ;      
				   Res8qq[][0] = res8_min ;        
				   Res8qq[][1] = res8_mean;        
				   Res8qq[][2] = res8_max ; 

				   

	  		     		 			 			     

						 Ajuste[0][] = pseudoR2LR; 	          
						 Ajuste[1][] = pseudoR2;	   
					

						 
						 
						  

									 

					     
	println ("\nTamanho da amostra.: ", cn);                                                                                                                                               
	println("\n","%11.1f","\n\n\t","%r",{"pseudoR2", "pseudoR2c"}, "%c", Ajuste);                                                                                                                                            
			  	                                                                                                                                
	println("\n    $\phi$ max     $\phi$ min      Grau de  Heteroscedasticidade estimado", maxc(phihat)~minc(phihat)~ maxc(phihat) /minc(phihat));            

					                                                                                                                                              
                                                                                                                                                                                          
			  		           
	fprint(fpout,"\n    $\phi$ max     $\phi$ min      Grau de  Heteroscedasticidade estimado", maxc(phihat)~minc(phihat)~ maxc(phihat) /minc(phihat));                                  
	fprint(fpout,"\n","%11.1f","\n\n\t","%r",{"pseudoR2", "pseudoR2c", "pseudoR2LR", "pseudoR2LRc","AIC","BIC" }, "%c", Ajuste);
					 
    fprint(fpout1,"\n  ", "%8.3f",Res1qq~Res2qq~Res3qq~Res4qq~Res5qq~Res6qq~Res7qq~Res8qq);     
    fprint(fpout2,"%9.4f",  res1_inf~res1_sup~res2_inf~res2_sup~res3_inf~res3_sup~res4_inf~res4_sup~res5_inf~res5_sup~res6_inf~res6_sup~res7_inf~res7_sup~res8_inf~res8_sup);                                                                                                                                          
    fprint(fpout16,"\n  ", "%8.3f",Res4qq~Res5qq~Res6qq);
	fprint(fpout17,"\n", "%12.8f",etahat);

//    /* Data, Hora e  Tempo ExecuÃ§Ã£o */											 //    /* Data, Hora e  Tempo de ExecuÃ§Ã£o */
       print( "\nDATE: ", date() );													       fprint(fpout, "\nDATE: ", date() );
       print( "\nTIME: ", time(), "\n" );											       fprint(fpout,"\nTIME: ", time(), "\n" );
       print( "\nTOTAL EXECUTION TIME: ", timespan(dExecTime) );					       fprint(fpout,"\nTOTAL EXECUTION TIME: ", timespan(dExecTime) );
       print( "\n" );																       fprint(fpout,"\n" );
  
  }
	 
