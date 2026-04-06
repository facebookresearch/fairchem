import {
  require_t4_templating
} from "/build/_shared/chunk-VVE4QZ3U.js";
import {
  require_csharp
} from "/build/_shared/chunk-JWAGNOMV.js";
import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/refractor/lang/t4-cs.js
var require_t4_cs = __commonJS({
  "../../node_modules/refractor/lang/t4-cs.js"(exports, module) {
    var refractorT4Templating = require_t4_templating();
    var refractorCsharp = require_csharp();
    module.exports = t4Cs;
    t4Cs.displayName = "t4Cs";
    t4Cs.aliases = [];
    function t4Cs(Prism) {
      Prism.register(refractorT4Templating);
      Prism.register(refractorCsharp);
      Prism.languages.t4 = Prism.languages["t4-cs"] = Prism.languages["t4-templating"].createT4("csharp");
    }
  }
});

export {
  require_t4_cs
};
//# sourceMappingURL=/build/_shared/chunk-DY2MRAPI.js.map
