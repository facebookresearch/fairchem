## Understanding preprocessed data 
 - ID index (example: 3589)
 - bulk_id: Materials Project ID of the bulk system used corresponding the the catalyst surface
 - bulk_symbols: Chemical composition of the bulk counterpart
 - miller_index: Describes how the bulk crystal was "sliced" to create the surface
 - traj_id: A unique hash string identifying the specific computational simulation (DFT trajectory) associated with this system.
 - slab_sid: Identifier associated with the corresponding slab (if available)
 - ads_symbols: Chemical composition of the adsorbate counterpart (adosrbate+slabs only)
 - nads: Number of adsorbates present
example:  "37864": {
    "bulk_id": "mp-19421",
    "miller_index": [
      1,
      1,
      0
    ],
    "nads": 0,
    "traj_id": "FeWO4_mp-19421_clean_Sl5m8ftNmX",
    "bulk_symbols": "Fe2W2O8" }
## After refining TiO2 from metadata
 - Including 8 results having bulk_symbols Ti-O (ratio 1:2)
 - 3 structures clean (don't have ads -> the first frame after cutting)
 - 5 structures have ads (O,C,H,CO,N)
 - All of structures in here can be understood as a beginning structure (clean structure) and the after structures 
  + For instance: 
    "3589": {
    "bulk_id": "mp-554278",
    "miller_index": [
      1,
      1,
      1
    ],
    "nads": 0,
    "traj_id": "TiO2_mp-554278_clean_AK6wavRPLH",
    "bulk_symbols": "Ti8O16" } => this is beginning structure because of a clean TiO2 which don't have any ads
  + Following this, The next structure will be
    "23315": {
    "bulk_id": "mp-554278",
    "miller_index": [
      1,
      1,
      1
    ],
    "nads": 1,
    "traj_id": "TiO2_mp-554278_AK6wavRPLH_4MPyNaaMJG",
    "bulk_symbols": "Ti8O16",
    "slab_sid": 3589,
    "ads_symbols": "CO"} => At the moment, CO which is a ads, linked with TiO2.
## Visualization
 - the results of visualization are 1218 frame of results
 - The information of this data including
  + y: total energy
  + pos: 3D coordinates ([100,3]=>[100 atoms and 3D structure])
  + cell: the size of box which is contained all of atoms
  + atomic_numbers: Z or proton number
  + natoms: the number of atoms and slabs
  + force: the force impacts individual atoms ([100,3] the force affects 3 directions, in there saving 100 atom data)
  + fixed: the fixed atom in bottom
  + tags: classify sub-surface (0), surface (1), adsorbate (2)
  + nads: Number of Adsorbate Atoms
  + sid: system id
  + fid: frame id 


