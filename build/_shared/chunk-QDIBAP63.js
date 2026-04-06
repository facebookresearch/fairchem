import {
  insertEdge,
  insertEdgeLabel,
  markers_default,
  positionEdgeLabel
} from "/build/_shared/chunk-QZO55VED.js";
import {
  insertCluster,
  insertNode,
  labelHelper
} from "/build/_shared/chunk-WP6UWRBG.js";
import {
  interpolateToCurve
} from "/build/_shared/chunk-WEDXSMQA.js";
import {
  common_default,
  getConfig
} from "/build/_shared/chunk-463FZTZ7.js";
import {
  __name,
  log
} from "/build/_shared/chunk-7PDOACA3.js";

// ../../node_modules/mermaid/dist/chunks/mermaid.core/chunk-N4CR4FBY.mjs
var internalHelpers = {
  common: common_default,
  getConfig,
  insertCluster,
  insertEdge,
  insertEdgeLabel,
  insertMarkers: markers_default,
  insertNode,
  interpolateToCurve,
  labelHelper,
  log,
  positionEdgeLabel
};
var layoutAlgorithms = {};
var registerLayoutLoaders = /* @__PURE__ */ __name((loaders) => {
  for (const loader of loaders) {
    layoutAlgorithms[loader.name] = loader;
  }
}, "registerLayoutLoaders");
var registerDefaultLayoutLoaders = /* @__PURE__ */ __name(() => {
  registerLayoutLoaders([
    {
      name: "dagre",
      loader: /* @__PURE__ */ __name(async () => await import("/build/_shared/dagre-6UL2VRFP-Z5LQKEJI.js"), "loader")
    },
    ...true ? [
      {
        name: "cose-bilkent",
        loader: /* @__PURE__ */ __name(async () => await import("/build/_shared/cose-bilkent-S5V4N54A-ATQLGFTK.js"), "loader")
      }
    ] : []
  ]);
}, "registerDefaultLayoutLoaders");
registerDefaultLayoutLoaders();
var render = /* @__PURE__ */ __name(async (data4Layout, svg) => {
  if (!(data4Layout.layoutAlgorithm in layoutAlgorithms)) {
    throw new Error(`Unknown layout algorithm: ${data4Layout.layoutAlgorithm}`);
  }
  const layoutDefinition = layoutAlgorithms[data4Layout.layoutAlgorithm];
  const layoutRenderer = await layoutDefinition.loader();
  return layoutRenderer.render(data4Layout, svg, internalHelpers, {
    algorithm: layoutDefinition.algorithm
  });
}, "render");
var getRegisteredLayoutAlgorithm = /* @__PURE__ */ __name((algorithm = "", { fallback = "dagre" } = {}) => {
  if (algorithm in layoutAlgorithms) {
    return algorithm;
  }
  if (fallback in layoutAlgorithms) {
    log.warn(`Layout algorithm ${algorithm} is not registered. Using ${fallback} as fallback.`);
    return fallback;
  }
  throw new Error(`Both layout algorithms ${algorithm} and ${fallback} are not registered.`);
}, "getRegisteredLayoutAlgorithm");

export {
  registerLayoutLoaders,
  render,
  getRegisteredLayoutAlgorithm
};
//# sourceMappingURL=/build/_shared/chunk-QDIBAP63.js.map
