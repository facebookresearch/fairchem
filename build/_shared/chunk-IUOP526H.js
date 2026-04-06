import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/refractor/lang/editorconfig.js
var require_editorconfig = __commonJS({
  "../../node_modules/refractor/lang/editorconfig.js"(exports, module) {
    module.exports = editorconfig;
    editorconfig.displayName = "editorconfig";
    editorconfig.aliases = [];
    function editorconfig(Prism) {
      Prism.languages.editorconfig = {
        // https://editorconfig-specification.readthedocs.io
        comment: /[;#].*/,
        section: {
          pattern: /(^[ \t]*)\[.+\]/m,
          lookbehind: true,
          alias: "selector",
          inside: {
            regex: /\\\\[\[\]{},!?.*]/,
            // Escape special characters with '\\'
            operator: /[!?]|\.\.|\*{1,2}/,
            punctuation: /[\[\]{},]/
          }
        },
        key: {
          pattern: /(^[ \t]*)[^\s=]+(?=[ \t]*=)/m,
          lookbehind: true,
          alias: "attr-name"
        },
        value: {
          pattern: /=.*/,
          alias: "attr-value",
          inside: {
            punctuation: /^=/
          }
        }
      };
    }
  }
});

export {
  require_editorconfig
};
//# sourceMappingURL=/build/_shared/chunk-IUOP526H.js.map
