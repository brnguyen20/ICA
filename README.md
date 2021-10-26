In this project, I applied Independent Component Analysis (ICA) - a signal decomposition technique - to a peer-generated database
of audio mixtures. One limitation is that, in order to use ICA, for n number of sources used to create a mixture, you must provide n mixtures to the ICA algorithm. 

For example, if a mixture is created using two separate sounds (i.e., a human talking and a bird chirping) , then you must provide two separate mixtures containing a human talking and a bird chirping (each at different amplitudes) for ICA to work.

In the "input files" folder, there are 180 mixtures containing two separate sounds - each of those mixtures was duplicated using the same two sounds, just at different levels. Therefore, there are 360 mixtures total.

**In the "output files" folder, you will find files in the format 'out{x}\_{y}.wav', where x is either 0 or 1, and represents one of the original signals used to create mixture number y. Therefore, for every 'out{0}\_{y}.wav' there is a 'out{1}\_{y}.wav' to account for both original signals.**
