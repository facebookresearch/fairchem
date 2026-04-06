import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/highlight.js/lib/languages/golo.js
var require_golo = __commonJS({
  "../../node_modules/highlight.js/lib/languages/golo.js"(exports, module) {
    function golo(hljs) {
      return {
        name: "Golo",
        keywords: {
          keyword: "println readln print import module function local return let var while for foreach times in case when match with break continue augment augmentation each find filter reduce if then else otherwise try catch finally raise throw orIfNull DynamicObject|10 DynamicVariable struct Observable map set vector list array",
          literal: "true false null"
        },
        contains: [
          hljs.HASH_COMMENT_MODE,
          hljs.QUOTE_STRING_MODE,
          hljs.C_NUMBER_MODE,
          {
            className: "meta",
            begin: "@[A-Za-z]+"
          }
        ]
      };
    }
    module.exports = golo;
  }
});
export default require_golo();
//# sourceMappingURL=/build/_shared/golo-YGMQZ22W.js.map
