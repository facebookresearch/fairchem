import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/highlight.js/lib/languages/mojolicious.js
var require_mojolicious = __commonJS({
  "../../node_modules/highlight.js/lib/languages/mojolicious.js"(exports, module) {
    function mojolicious(hljs) {
      return {
        name: "Mojolicious",
        subLanguage: "xml",
        contains: [
          {
            className: "meta",
            begin: "^__(END|DATA)__$"
          },
          // mojolicious line
          {
            begin: "^\\s*%{1,2}={0,2}",
            end: "$",
            subLanguage: "perl"
          },
          // mojolicious block
          {
            begin: "<%{1,2}={0,2}",
            end: "={0,1}%>",
            subLanguage: "perl",
            excludeBegin: true,
            excludeEnd: true
          }
        ]
      };
    }
    module.exports = mojolicious;
  }
});
export default require_mojolicious();
//# sourceMappingURL=/build/_shared/mojolicious-HGKXBKLS.js.map
