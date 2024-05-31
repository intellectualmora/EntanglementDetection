clear,clc

qubit=4;
sam=1*10^4;

[Ix,Iy,Iz,IHx,IHy,IHz,sIHz] = prodop(1/2,qubit);
I{1}=Ix(:,:,:);I{2}=Iy(:,:,:);I{3}=Iz(:,:,:);

state0=zeros(2^qubit,1);state0(1,1)=1;
state0=expm(-1i*pi/8*IHz)*expm(-1i*pi/8*IHy)*state0;

T_list=pi/50:pi/50:2*pi;

aveg_vqe=zeros(sam,qubit,3,50);
S=zeros(sam,3,100);
tic
for mm=1:sam
    
    JJ=ones(qubit-1,1)*(2*rand-1);
    gg=ones(qubit,1)*(2*rand-1);
    Ham=zeros(2^qubit,2^qubit);
    for ii=1:qubit-1
        Ham=Ham+JJ(ii)*Iz(:,:,ii)*2*Iz(:,:,ii+1)*2;
    end
    for ii=1:qubit
        Ham=Ham+gg(ii)*Ix(:,:,ii)*2;
    end
    
    
    for dd=1:length(T_list)
        tau=T_list(dd);
        statef=expm(-1i*Ham*tau)*state0;
        
        if dd<=50
            for xx=1:qubit
                for yy=1:3
                    aveg_vqe(mm,xx,yy,dd)=real(statef'*I{yy}(:,:,xx)*2*statef);
                end
            end
        end
        
        
        rhosub=Partrace(statef*statef',[2:qubit]);
        S(mm,1,dd)=-real(log2(trace(rhosub^2)));
        rhosub=Partrace(statef*statef',[3:qubit]);
        S(mm,2,dd)=-real(log2(trace(rhosub^2)));
        rhosub=Partrace(statef*statef',[4:qubit]);
        S(mm,3,dd)=-real(log2(trace(rhosub^2)));
    end
end
inputdata=aveg_vqe;
outdata=S;

toc
save Evaldata_4qubit_Ham_dynamics_simpleHam.mat inputdata outdata