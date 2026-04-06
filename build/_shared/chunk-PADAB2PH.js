import {
  require_vbnet
} from "/build/_shared/chunk-XZPFVGSQ.js";
import {
  require_t4_templating
} from "/build/_shared/chunk-VVE4QZ3U.js";
import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/refractor/lang/t4-vb.js
var require_t4_vb = __commonJS({
  "../../node_modules/refractor/lang/t4-vb.js"(exports, module) {
    var refractorT4Templating = require_t4_templating();
    var refractorVbnet = require_vbnet();
    module.exports = t4Vb;
    t4Vb.displayName = "t4Vb";
    t4Vb.aliases = [];
    function t4Vb(Prism) {
      Prism.register(refractorT4Templating);
      Prism.register(refractorVbnet);
      Prism.languages["t4-vb"] = Prism.languages["t4-templating"].createT4("vbnet");
    }
  }
});

export {
  require_t4_vb
};
//# sourceMappingURL=/build/_shared/chunk-PADAB2PH.js.map
