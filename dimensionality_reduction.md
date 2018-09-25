<_sfiguser> hello all, what do you suggest to visualize a multidimensional clustering result ? 
<_sfiguser> i mean can i proceed with a PCA to two dimensions and then coloring points dependeing on the output cluster ? 
* Gustavo6046 has quit (Ping timeout: 260 seconds)
* Gustavo6056 is now known as Gustavo6046
<_sfiguser> is it a good idea? how do people generally proceed for this task ? 
<_sfiguser> i also saw somebody used MDS or t-SNE, but I am not sure they can be used in this context
<_sfiguser> any idea ?
<RandIter> Why can't t-SNE be used?
* libertyprime (~libertypr@30.84.69.111.dynamic.snap.net.nz) has joined
* mauz555 (~jimmy@2a01:e35:8ab1:dea0:dcc3:659c:6cb8:83b7) has joined
* Mia has quit (Ping timeout: 244 seconds)
* Mia (~Mia@217.131.89.212) has joined
* Mia has quit (Changing host)
* Mia (~Mia@unaffiliated/mia) has joined
* cerbere has quit (Remote host closed the connection)
* dbff2 (~dbff2@guest-wireless-207-151-035-005.usc.edu) has joined
* silver_ (~silver@93.84.17.11) has joined
* derk0pf (~derk0pf@178.115.131.26.wireless.dyn.drei.com) has joined
* hckiang has quit (Ping timeout: 252 seconds)
* cryptocat has quit (Ping timeout: 245 seconds)
* tirohia (~ben@123.100.102.236) has joined
* silver has quit (Ping timeout: 240 seconds)
<brand0> _sfiguser, t-SNE is what I would use
<brand0> _sfiguser, and to your second questions, you'd make the output the color variable and the inputs down to x, y
<_sfiguser> ok brand0 but why t-SNE instead of MDS ? 
<_sfiguser> RandIter, why t-SNE instead of MDS ? 
* rcdilorenzo (~rcdiloren@cpe-24-163-97-97.nc.res.rr.com) has joined
* Orpheon has quit (Remote host closed the connection)
<brand0> _sfiguser, you should try both, but t-SNE tries to exaggerate groups of data points, MDS will try to preserve the natural structure of the data
* quant4242 has quit (Quit: quant4242)
* astronavt_ (~astronavt@cpe-74-71-190-13.nyc.res.rr.com) has joined
<_sfiguser> brand0, ok is there any book/course explaining these techniques ? 
<brand0> _sfiguser, wiki pages are pretty good. you should get familiar with PCA via tutorials as an entry point to dimension reduction
<_sfiguser> brand0,  ok once i tryt both how can i choose which one is a better representation? 
<brand0> there's lots of online tutorials about t-SNE in particular
* tobiasu22 (~tobiasu@78.160.202.148) has joined
* tobiasu22 has quit (Remote host closed the connection)
* astronavt has quit (Read error: Connection reset by peer)
<brand0> _sfiguser, just by how it looks. just consider the question: do I want data to look more like clusters, or do I want to see data more scattered according to how similar they are individually
* astronavt__ (~astronavt@8.37.179.150) has joined
<brand0> if you want the first: t-SNE, second MDS
* Douhet has quit (Ping timeout: 240 seconds)
<_sfiguser> brand0, sorry i don't understand the first question...
<_sfiguser> "do i want more clusters?" 
<_sfiguser> well i choose clusters before applying t-SNE and MDS
<_sfiguser> brand0, 
* Douhet (~Douhet@unaffiliated/douhet) has joined
<brand0> _sfiguser, yeah, but your output (class) variable will just be turned into a color
<brand0> so your given classes aren't going to be taken into consideration with unsupervised dimension reduction
<_sfiguser> which one between the two is unsupervised dimension reduction? 
<_sfiguser> brand0, sorry can you repeat the two questions by being more precise and with more details ? sorry i'm a noob
<_sfiguser> i'm making confusion between the two questions...
<_sfiguser> so let's say i have a 10 dimensional dataset and i determined by using a silhoutte analysis that a good number of clusters is 4
<brand0> I was answering your "well I chose clusters ..."
* astronavt_ has quit (Ping timeout: 272 seconds)
<_sfiguser> now i use t-SNE and MDS to visualize data in 2D
<_sfiguser> how do the questions fit into this ? 
* smccarthy has quit ()
<brand0> _sfiguser, so what you'll get by running t-SNE on it is 10D input goes to 2D x, y
<brand0> then you can color your points by the class you selected
<brand0> if you chose a good number of clusters, you'll get four clusters from t-SNE and they'll all be the same color
<brand0> more likely you'll see lots of little groups of data
<_sfiguser> ok brand0 what about MDS ? 
<brand0> MDS will get you more of a grid with the points scattered through them. if your clustering is good then you should have groups (by color) of points with possibly some more bare places between them
<brand0> it's hard to tell, I'm only guessing on what the data *might* look like
<brand0> you really just need to try both
<brand0> scikit-learn can do them both, you just drop-in TSNE or MDS
<brand0> everything else is identical
<_sfiguser> brand0, ok so this is a common way to proceed with a clustering problem ? 
<brand0> _sfiguser, it's a common way to *visualize* your clusters
<brand0> and visualize your data, in general. sometimes it's a way to guess how many clusters you should use


#
I can either visualize my data to infer clusters, or
once clustering is finished, i can visualize clustering
by using t-SNE and MDS, t-SNE exaggerates clusters, MDS
tries to preserve original data structure
# 



