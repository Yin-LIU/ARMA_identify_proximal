h = 10;  % predicted days

stock = readmatrix("Netflix_stock_history.csv");
stock = stock(:,2);
stock = diff(stock);

stock_train = stock(1:end-h);
stock_test = stock(end-h+1:end);

% stock_input = diff(stock_train); % take difference 
[stock_input, mu,sigma] = zscore(stock_train);

[result,AIC, BIC,X_all,const_all,ic] = est_HSARMA(stock_input);




%% MSE for ARIMA
Mdl = arima(5,0,0);
Mdl.Variance = 1;
[EstMdl,~,LogL]=estimate(Mdl,stock_input);
[A_ARMA,B_ARMA] =  aicbic(LogL,EstMdl.P+EstMdl.Q+1,length(stock_input));
pred_ARMA = forecast(EstMdl,h,stock_input);


pred_true = (stock_test - mu)/sigma;
MSE_ARMA =((pred_true - pred_ARMA).^2);

%% MSE for HSARMA
i=4;j=4;
Mdl_H = arima('AR',X_all{i,j}(1:14),'MA',X_all{i,j}(15:28),'Constant',const_all{i,j},'Variance',1);
[EstMdl_H,~,LogL] = estimate(Mdl_H,stock_input);
[A_H,B_H]=aicbic(LogL,EstMdl_H.P+EstMdl_H.Q+1,length(stock_input));
pred_H = forecast(EstMdl_H,h,stock_input);

MSE_H = ((pred_true - pred_H).^2);



%% plot heat map
AICC = zeros(10,10);
for i = 1:10
    for j=1:10
        AICC(i,j)=ic{i,j}.aicc;
    end
end
figure(1)
h1=heatmap(round(logspace(log10(100),log10(1),10),1),round(logspace(log10(100),log10(1),10),1),AIC);
h1.Title='AIC';
h1.YLabel = '\lambda_{AR}';
h1.XLabel='\lambda_{MA}';
h1.ColorbarVisible = 'off';

figure(2)
h2=heatmap(round(logspace(log10(100),log10(1),10),1),round(logspace(log10(100),log10(1),10),1),AICC);
h2.Title = 'AICc';
h2.YLabel = '\lambda_{AR}';
h2.XLabel='\lambda_{MA}';
h2.ColorbarVisible = 'off';


figure(3)
h2=heatmap(round(logspace(log10(100),log10(1),10),1),round(logspace(log10(100),log10(1),10),1),BIC);
h2.Title='BIC';
h2.YLabel = '\lambda_{AR}';
h2.XLabel='\lambda_{MA}';
h2.ColorbarVisible = 'off';



%%  Plot AIC
fig = figure(1);
clf;
ax=axes(fig);
h=imagesc(ax,AIC);
set(ax, 'XTickLabel',round(logspace(log10(100),log10(1),10),1), 'YTickLabel',round(logspace(log10(100),log10(1),10),1))
title('AIC')
ylabel('\lambda_{MA}')
xlabel('\lambda_{AR}')

n = 256;
cmap = [linspace(.9,0,n)', linspace(.9447,.447,n)', linspace(.9741,.741,n)'];
colormap(ax,cmap)

hold on
% Set grid lines
arrayfun(@(x)xline(ax,x,'k-','Alpha',1),0.5:1:10.5)
arrayfun(@(y)yline(ax,y,'k-','Alpha',1),0.5:1:10.5)



[xTxt, yTxt] = ndgrid(1:10,1:10);

labels = cell(10,10);

for i =1:10
    for j = 1:10
        temp = num2str(AIC(i,j),'%.0f');
        P = find(X_all{i,j}(1:14),1,'last');
        Q = find(X_all{i,j}(15:end),1,'last');
        if isempty(P)
            P = 0;
        end
        if isempty(Q)
            Q=0;
        end
        
        labels{i,j} =sprintf(num2str(AIC(i,j),'%.0f')+ "\n("+num2str(P)+","+num2str(Q)+")");
    end
end
labels=labels';
th = text(xTxt(:), yTxt(:), (labels(:)), ...
    'VerticalAlignment', 'middle','HorizontalAlignment','Center');

hold off
%%  Plot AICc
fig = figure(2);
ax=axes(fig);
h=imagesc(ax,AICC);
set(ax, 'XTickLabel',round(logspace(log10(100),log10(1),10),1), 'YTickLabel',round(logspace(log10(100),log10(1),10),1))
title('AICc')
ylabel('\lambda_{MA}')
xlabel('\lambda_{AR}')

n = 256;
cmap = [linspace(.9,0,n)', linspace(.9447,.447,n)', linspace(.9741,.741,n)'];
colormap(ax,cmap)

hold on
% Set grid lines
arrayfun(@(x)xline(ax,x,'k-','Alpha',1),0.5:1:10.5)
arrayfun(@(y)yline(ax,y,'k-','Alpha',1),0.5:1:10.5)



[xTxt, yTxt] = ndgrid(1:10,1:10);

labels = cell(10,10);

for i =1:10
    for j = 1:10
        P = find(X_all{i,j}(1:14),1,'last');
        Q = find(X_all{i,j}(15:end),1,'last');
        if isempty(P)
            P = 0;
        end
        if isempty(Q)
            Q=0;
        end
        labels{i,j} =sprintf(num2str(AICC(i,j),'%.0f')+ "\n("+num2str(P)+","+num2str(Q)+")");
    end
end
labels=labels';
th = text(xTxt(:), yTxt(:), (labels(:)), ...
    'VerticalAlignment', 'middle','HorizontalAlignment','Center');
hold off

%%  Plot BIC
fig = figure(3);
ax=axes(fig);
h=imagesc(ax,BIC);
set(ax, 'XTickLabel',round(logspace(log10(100),log10(1),10),1), 'YTickLabel',round(logspace(log10(100),log10(1),10),1))
title('BIC')
ylabel('\lambda_{MA}')
xlabel('\lambda_{AR}')

n = 256;
cmap = [linspace(.9,0,n)', linspace(.9447,.447,n)', linspace(.9741,.741,n)'];
colormap(ax,cmap)

hold on
% Set grid lines
arrayfun(@(x)xline(ax,x,'k-','Alpha',1),0.5:1:10.5)
arrayfun(@(y)yline(ax,y,'k-','Alpha',1),0.5:1:10.5)



[xTxt, yTxt] = ndgrid(1:10,1:10);

labels = cell(10,10);

for i =1:10
    for j = 1:10
        temp = num2str(BIC(i,j),'%.0f');
        P = find(X_all{i,j}(1:14),1,'last');
        Q = find(X_all{i,j}(15:end),1,'last');
        if isempty(P)
            P = 0;
        end
        if isempty(Q)
            Q=0;
        end
        
        labels{i,j} =sprintf(num2str(BIC(i,j),'%.0f')+ "\n("+num2str(P)+","+num2str(Q)+")");
    end
end
labels=labels';
th = text(xTxt(:), yTxt(:), (labels(:)), ...
    'VerticalAlignment', 'middle','HorizontalAlignment','Center');
hold off
