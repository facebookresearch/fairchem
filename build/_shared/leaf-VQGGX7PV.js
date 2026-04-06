import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/highlight.js/lib/languages/leaf.js
var require_leaf = __commonJS({
  "../../node_modules/highlight.js/lib/languages/leaf.js"(exports, module) {
    function leaf(hljs) {
      return {
        name: "Leaf",
        contains: [
          {
            className: "function",
            begin: "#+[A-Za-z_0-9]*\\(",
            end: / \{/,
            returnBegin: true,
            excludeEnd: true,
            contains: [
              {
                className: "keyword",
                begin: "#+"
              },
              {
                className: "title",
                begin: "[A-Za-z_][A-Za-z_0-9]*"
              },
              {
                className: "params",
                begin: "\\(",
                end: "\\)",
                endsParent: true,
                contains: [
                  {
                    className: "string",
                    begin: '"',
                    end: '"'
                  },
                  {
                    className: "variable",
                    begin: "[A-Za-z_][A-Za-z_0-9]*"
                  }
                ]
              }
            ]
          }
        ]
      };
    }
    module.exports = leaf;
  }
});
export default require_leaf();
//# sourceMappingURL=/build/_shared/leaf-VQGGX7PV.js.map
