- Converting the problem into a linear algebra problem:
  - \( S_{gi} = S_0 \Sigma \left(\sum_{l=1}^{N} [ w_l \exp(-b \mathbf{g}_i^T \mathbf{D}_l^{-1} \mathbf{g}_i)]\right) \)
  - Where \( w \) is a sparse vector
  - \( \mathbf{g}_i \) is the gradient direction
  - \( b \) is the b-value (a constant scalar)
  - \( S_{gi} \) is the signal intensity
  - \( S_0 \) is the signal intensity in the absence of diffusion sensitization
  - \( \mathbf{D}_l \) is the diffusion tensor, \( \mathbf{D}_l = \mathbf{V}_l \mathbf{L} \mathbf{v}_l^T \)
    - Where \( \mathbf{V}_l = (\mathbf{v}_{l1}|\mathbf{v}_{l2}|\mathbf{v}_{l3}) \)
    - \( \mathbf{v}_{l1} \) is chosen randomly on a uniform sphere (discrete)
    - \( \mathbf{v}_{l2} \) is chosen randomly on the sphere, from the plane perpendicular to \( \mathbf{v}_{l1} \)
    - \( \mathbf{v}_{l3} \) is the cross product of the two

Given:
- \( S_{gi} \), signal intensity
- \( S_0 \), signal intensity in the absence of diffusion sensitization
- \( \mathbf{g}_i \), gradient direction
- \( b \), b-value
- \( N \), number of diffusion tensors

To initialize:
- \( \lambda \), the eigenvalues of the diffusion tensor
- \( \mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3 \), the eigenvectors of the diffusion tensor
  - @assumption: \( \lambda \)s are fixed for all the diffusion tensors, and only the weights change

Dimensions:
- \( m \), number of gradient directions
- \( N \), number of diffusion tensors

To evaluate:
- \( \mathbf{D}_l \), matrix of eigenvalues
- \( \Sigma_l \), Diffusion
- \( w \), weights

Now, to the linear problem:
- \( \mathbf{y} = \mathbf{A}\mathbf{w} \)
- \( \mathbf{y} = \frac{S_{gi}}{S_0} \), of dimension \( m \times 1 \)
- \( \mathbf{A} \), matrix such that \( A_{il} = \exp(-b \mathbf{g}_i^T \mathbf{D}_l^{-1} \mathbf{g}_i) \)
- And \( \mathbf{w} \) is the vector of weights

Finding the best lambdas:
- Fix three lambdas, in descending order
- Around the range of lambdas, do a grid search
- Then what?
- Something about lambdas fixed for all diffusion tensors
