# Architecture Thoughts
Initial thoughts on architecture of network_wrangler, lasso, ranch (and travel-model-two-networks)

## Code References
Note that the below diagrams and text link to MTC's working versions of these modules, which are:
* [BayAreaMetro/network_wrangler, generic_agency branch](https://github.com/BayAreaMetro/network_wrangler/tree/generic_agency)
* [BayAreaMetro/Lasso, mtc_parameters branch](https://github.com/BayAreaMetro/Lasso/tree/mtc_parameters)
* [wsp-sag/Ranch](https://github.com/wsp-sag/Ranch) is not being used by our codebase at this time; instead we have pipeline scripts in
* [BayAreaMetro/travel-model-two-networks, develop branch](https://github.com/BayAreaMetro/travel-model-two-networks/tree/develop)

## Overall Comments
I think these libraries are a great start but there are some issues that I see as being critical to fix:
* **Documentation** -- Whenever our staff looks at the code, we end up with *so many* questions.  We'd be happy to add answers to the documentation as we figure it out BUT they won't be useful to the greater project without
* **Branch housekeeping** -- There are many active branches: 
  * [wsp-sag/network_wranger](https://github.com/wsp-sag/network_wrangler/branches): master, develop, develop_with_ranch, generric_agency
  * [wsp-sag/Lasso](https://github.com/wsp-sag/Lasso/branches): master, develop, develop_with_ranch, generic_agency
  * Not to mention forks which add even more versions of these branches
  * The work to make a plan for this and execute it is likely related to the future Governance/Funding conversation
* **Tests** -- I think a lot of the potential for making sure these libraries work as expected with continuous development is through testing infrastructure, but this is currently [not maintained](https://travis-ci.org/github/wsp-sag/network_wrangler/branches).  See my attempt to fix tests on [wsp-sag/network_wrangler, develop branch](https://github.com/wsp-sag/network_wrangler/pull/281)

## Existing Classes

* [RoadwayNetwork](https://bayareametro.github.io/network_wrangler/_generated/network_wrangler.RoadwayNetwork/) is described simply as a "Representation of a Roadway Network".  What does this mean?  What are the required fields?  
  * It looks like there are some schemas defined in [network_wrangler/schemas](https://github.com/BayAreaMetro/network_wrangler/tree/generic_agency/network_wrangler/schemas) which I think are interesting and potentially useful, but I am not sure if they're being used?  I think they have value, especially in validation, but would also like to see the documentation of the class reflect them to make them easier to understand and use.


### Basic Network Classes
```mermaid
classDiagram 
RoadwayNetwork <-- TransitNetwork
RoadwayNetwork <|-- ModelRoadwayNetwork

%% network_wrangler classes

  class RoadwayNetwork {
    +GeoDataFrame links_df
    +GeoDataFrame nodes_df
    +GeoDataFrame shapes_df
    +int crs
    +str node_foreign_key
    +list~str~ link_foreign_key
    +str shape_foreign_key
    +list~str~ unique_link_ids
    +list~str~ unique_node_ids
    +dict modes_to_network_link_variables
    +dict modes_to_network_nodes_variables
    +int managed_lanes_node_id_scalar
    +int managed_lanes_link_id_scalar
    +list~str~ managed_lanes_required_attributes
    +list keep_same_attributes_ml_and_gp
    +dict selections
  }
  link RoadwayNetwork "https://bayareametro.github.io/network_wrangler/_generated/network_wrangler.RoadwayNetwork/" "network_wrangler.RoadwayNetwork"
  
  class TransitNetwork {
    +feed DotDict
    +config nx.DiGraph
    +road_net RoadwayNetwork
    +graph nx.MultiDiGrapp
  }
  link TransitNetwork "https://bayareametro.github.io/network_wrangler/_generated/network_wrangler.TransitNetwork/" "network_wrangler.TransitNetwork"
  
%% lasso classes

  class ModelRoadwayNetwork {
  }
  link ModelRoadwayNetwork "https://bayareametro.github.io/Lasso/_generated/lasso.ModelRoadwayNetwork/#lasso.ModelRoadwayNetwork" "lasso.ModelRoadwayNetwork"
```

### Project Card and Scenario
```mermaid
classDiagram 
  class ProjectCard {
  }
  link ProjectCard "https://bayareametro.github.io/network_wrangler/_generated/network_wrangler.ProjectCard/" "network_wrangler.ProjectCard"
  
  class Scenario {
  }
  link Scenario "https://bayareametro.github.io/network_wrangler/_generated/network_wrangler.Scenario/" "network_wrangler.Scenario"
```

### Lasso Classes
```mermaid
classDiagram
  class StandardTransit {
  }
  link StandardTransit "https://bayareametro.github.io/Lasso/_generated/lasso.StandardTransit/#lasso.StandardTransit" "lasso.StandardTransit"
  
  class CubeTransit {
  }
  link CubeTransit "https://bayareametro.github.io/Lasso/_generated/lasso.CubeTransit/#lasso.CubeTransit" "lasso.CubeTransit"

  class CubeTransformer {
  }
  link CubeTransformer "https://bayareametro.github.io/Lasso/_generated/lasso.CubeTransformer/#lasso.CubeTransformer" "lasso.CubeTransformer"

  class SetupEmme {
  }
  link SetupEmme "https://bayareametro.github.io/Lasso/_generated/lasso.SetupEmme/#lasso.SetupEmme" "lasso.SetupEmme"
```

## Other references
* [Mermaid documentation on Class Diagrams](https://mermaid-js.github.io/mermaid/#/classDiagram)
* [Understanding JSON Schema](https://json-schema.org/understanding-json-schema/index.html)

