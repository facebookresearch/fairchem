import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/refractor/lang/false.js
var require_false = __commonJS({
  "../../node_modules/refractor/lang/false.js"(exports, module) {
    module.exports = $false;
    $false.displayName = "$false";
    $false.aliases = [];
    function $false(Prism) {
      ;
      (function(Prism2) {
        Prism2.languages["false"] = {
          comment: {
            pattern: /\{[^}]*\}/
          },
          string: {
            pattern: /"[^"]*"/,
            greedy: true
          },
          "character-code": {
            pattern: /'(?:[^\r]|\r\n?)/,
            alias: "number"
          },
          "assembler-code": {
            pattern: /\d+`/,
            alias: "important"
          },
          number: /\d+/,
          operator: /[-!#$%&'*+,./:;=>?@\\^_`|~ßø]/,
          punctuation: /\[|\]/,
          variable: /[a-z]/,
          "non-standard": {
            pattern: /[()<BDO®]/,
            alias: "bold"
          }
        };
      })(Prism);
    }
  }
});

export {
  require_false
};
//# sourceMappingURL=/build/_shared/chunk-3I2WKTH3.js.map
