## FreeBall - Volleyball Analysis Tool

### Scope of Work
1. Ball location prediction \
    Inputs: Video Frame (360 x 640), Grayscale or Color not yet decided \
    Outputs: 2D Pixel Coordinates of ball location, estimated ball pixel radius

    Might not need machine learning if we are able to use background estimation to remove background and only look for circular-ish objects. Can use centroid tracker to keep tracking the ball while its in the frame.

2. Determining Ball 3D Velocity \
    Inputs: Temporal Group of Frames (360 x 640 x N) \
    Outputs: Heatmap showing ball velocity in same dimension as inputs

Considerations
- Ball might not always be circular, especially during a spike or when the ball is travelling quickly
- Large dataset, will need to perform batch loading for training of models, and also not able to preprocess and store data before hand. Will need to use tensorflow's Sequential to process the video as part of the model