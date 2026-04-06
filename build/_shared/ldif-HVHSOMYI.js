import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/highlight.js/lib/languages/ldif.js
var require_ldif = __commonJS({
  "../../node_modules/highlight.js/lib/languages/ldif.js"(exports, module) {
    function ldif(hljs) {
      return {
        name: "LDIF",
        contains: [
          {
            className: "attribute",
            begin: "^dn",
            end: ": ",
            excludeEnd: true,
            starts: {
              end: "$",
              relevance: 0
            },
            relevance: 10
          },
          {
            className: "attribute",
            begin: "^\\w",
            end: ": ",
            excludeEnd: true,
            starts: {
              end: "$",
              relevance: 0
            }
          },
          {
            className: "literal",
            begin: "^-",
            end: "$"
          },
          hljs.HASH_COMMENT_MODE
        ]
      };
    }
    module.exports = ldif;
  }
});
export default require_ldif();
//# sourceMappingURL=/build/_shared/ldif-HVHSOMYI.js.map
