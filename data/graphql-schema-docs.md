# GraphQL Schema Reference


## Query: fetchChannels
**Description:** Returns all channels available for a given location.
There are no required parameters.

**Arguments:** None


**Return Type:** [channel]

**Return Fields:**
- alternateNames
- channelDescription
- channelName
- genre
- ott
- popular
- tags
- topShows

## Query: fetchMarketCompetitors
**Description:** Returns competitors with available packages.

**Arguments:**
- where: marketCompetitorInput! (required)
  - zipCodes: String! (required) - 5-character zip code or a comma-separated list of zip codes. Example: "73034" or "73034,73102,73114"
  - productCategories: String (optional) - One more more comma-separated product categories. Example: "Internet, Voice"

**Return Type:** [competitor]

**Return Fields:**
- competitor
- productCategories

## Query: fetchPackageById
**Description:** Returns a single package by packageFactId.

**Arguments:**
- where: packageIdInput! (required)
  - packageFactId: Int! (required) - packageFactId identifier. Example: 1234567

**Return Type:** package

**Return Fields:**
- activationChargePromotional
- activationChargeStandard
- alternateRatesNotes
- additionalPromotionSet2Description
- additionalPromotionSet3Description
- additionalPromotionSet4Description
- additionalPromotionSet5Description
- addOnChannelPackages
- addOnChannelLineups
- advertisedChannelCount
- advertisedHDChannelCount
- advertisedVoiceFeatureCount
- baseChannelCount
- baseChannelLineups
- callingRateNotes
- cashBackGiftCard
- competitor
- contract
- convertedDownloadSpeed
- customerTypes
- dmaCode
- downloadSpeed
- downloadSpeedUnit
- dvrServiceFees
- earlyTerminationFee
- etfNotes
- flexDisclaimer
- hdServiceFees
- highSpeedHotspotDataIncludedMB
- highSpeedWirelessDataIncludedMB
- includedInternetEquipment
- includedVideoReceivers
- includedVoiceEquipment
- internetComponent
- internetPurchaseName
- internetPurchasePrice
- internetRentalName
- internetRentalPriceMonth1
- internetRentalStandardPrice
- internetTransmission
- internetUsageCap
- internetUsageCapUnit
- internetUsageOverageCharge
- language
- leadOffer
- localCallingRates
- localChannelFee
- longDistanceCallingRates
- multiRoomServiceFees
- optionalVideoReceivers
- otherIncentivePromotions
- packageFactId
- packageName
- priceStep1EndMonth
- priceStep1Price
- priceStep1StartMonth
- priceStep2EndMonth
- priceStep2Price
- priceStep2StartMonth
- priceStep3EndMonth
- priceStep3Price
- priceStep3StartMonth
- priceStep4EndMonth
- priceStep4Price
- priceStep4StartMonth
- priceStep5EndMonth
- priceStep5Price
- priceStep5StartMonth
- productCategory
- professionalInstallationChargePromotional
- professionalInstallationChargeStandard
- promotionDescription
- propertyTypes
- proprietary
- purchaseInternetEquipment
- receiverConfig1Description
- receiverConfig1Name
- receiverConfig1OneTimeFee
- receiverConfig1OneTimeFeeDescription
- receiverConfig1PriceMonth1
- receiverConfig1StandardPrice
- receiverConfig2Description
- receiverConfig2Name
- receiverConfig2OneTimeFee
- receiverConfig2OneTimeFeeDescription
- receiverConfig2PriceMonth1
- receiverConfig2StandardPrice
- receiverConfig3Description
- receiverConfig3Name
- receiverConfig3OneTimeFee
- receiverConfig3OneTimeFeeDescription
- receiverConfig3PriceMonth1
- receiverConfig3StandardPrice
- receiverConfig4Description
- receiverConfig4Name
- receiverConfig4OneTimeFee
- receiverConfig4OneTimeFeeDescription
- receiverConfig4PriceMonth1
- receiverConfig4StandardPrice
- regionalSportsFee
- rentalInternetEquipment
- rentalVoiceEquipment
- requiresBundling
- salesChannels
- selfInstallationChargePromotional
- selfInstallationChargeStandard
- standardMonthlyCharge
- subscriberLineCharge
- termCommitment
- totalHotspotDataIncludedMB
- uniqueWirelessCompetitiveClaims
- uploadSpeed
- uploadSpeedUnit
- videoComponent
- videoReceiverFeatures
- videoTransmission
- voiceComponent
- voiceTransmission
- wirelessCallingRates
- wirelessComponent
- wirelessDataIncludedMB
- wirelessDataRates
- wirelessLinesIncluded
- wirelessPlanType
- wirelessTextingRates
- maxFirstMonthlyWirelessBundleDiscountNote

## Query: fetchMyCurrentPackages
**Description:** Returns all proprietary packages for a location for all competitors.

**Arguments:**
- where: locationInput! (required)
  - zipCodes: String (optional) - 5-character zip code or a comma-separated list of zip codes. Example: "73034" or "73034,73102,73114"
  - city: String (optional) - Municipality name. Example: "Atlanta"
  - state: String (optional) - 2-character state or US territory code. Example: "GA"

**Return Type:** [package]

**Return Fields:**
- activationChargePromotional
- activationChargeStandard
- alternateRatesNotes
- additionalPromotionSet2Description
- additionalPromotionSet3Description
- additionalPromotionSet4Description
- additionalPromotionSet5Description
- addOnChannelPackages
- addOnChannelLineups
- advertisedChannelCount
- advertisedHDChannelCount
- advertisedVoiceFeatureCount
- baseChannelCount
- baseChannelLineups
- callingRateNotes
- cashBackGiftCard
- competitor
- contract
- convertedDownloadSpeed
- customerTypes
- dmaCode
- downloadSpeed
- downloadSpeedUnit
- dvrServiceFees
- earlyTerminationFee
- etfNotes
- flexDisclaimer
- hdServiceFees
- highSpeedHotspotDataIncludedMB
- highSpeedWirelessDataIncludedMB
- includedInternetEquipment
- includedVideoReceivers
- includedVoiceEquipment
- internetComponent
- internetPurchaseName
- internetPurchasePrice
- internetRentalName
- internetRentalPriceMonth1
- internetRentalStandardPrice
- internetTransmission
- internetUsageCap
- internetUsageCapUnit
- internetUsageOverageCharge
- language
- leadOffer
- localCallingRates
- localChannelFee
- longDistanceCallingRates
- multiRoomServiceFees
- optionalVideoReceivers
- otherIncentivePromotions
- packageFactId
- packageName
- priceStep1EndMonth
- priceStep1Price
- priceStep1StartMonth
- priceStep2EndMonth
- priceStep2Price
- priceStep2StartMonth
- priceStep3EndMonth
- priceStep3Price
- priceStep3StartMonth
- priceStep4EndMonth
- priceStep4Price
- priceStep4StartMonth
- priceStep5EndMonth
- priceStep5Price
- priceStep5StartMonth
- productCategory
- professionalInstallationChargePromotional
- professionalInstallationChargeStandard
- promotionDescription
- propertyTypes
- proprietary
- purchaseInternetEquipment
- receiverConfig1Description
- receiverConfig1Name
- receiverConfig1OneTimeFee
- receiverConfig1OneTimeFeeDescription
- receiverConfig1PriceMonth1
- receiverConfig1StandardPrice
- receiverConfig2Description
- receiverConfig2Name
- receiverConfig2OneTimeFee
- receiverConfig2OneTimeFeeDescription
- receiverConfig2PriceMonth1
- receiverConfig2StandardPrice
- receiverConfig3Description
- receiverConfig3Name
- receiverConfig3OneTimeFee
- receiverConfig3OneTimeFeeDescription
- receiverConfig3PriceMonth1
- receiverConfig3StandardPrice
- receiverConfig4Description
- receiverConfig4Name
- receiverConfig4OneTimeFee
- receiverConfig4OneTimeFeeDescription
- receiverConfig4PriceMonth1
- receiverConfig4StandardPrice
- regionalSportsFee
- rentalInternetEquipment
- rentalVoiceEquipment
- requiresBundling
- salesChannels
- selfInstallationChargePromotional
- selfInstallationChargeStandard
- standardMonthlyCharge
- subscriberLineCharge
- termCommitment
- totalHotspotDataIncludedMB
- uniqueWirelessCompetitiveClaims
- uploadSpeed
- uploadSpeedUnit
- videoComponent
- videoReceiverFeatures
- videoTransmission
- voiceComponent
- voiceTransmission
- wirelessCallingRates
- wirelessComponent
- wirelessDataIncludedMB
- wirelessDataRates
- wirelessLinesIncluded
- wirelessPlanType
- wirelessTextingRates
- maxFirstMonthlyWirelessBundleDiscountNote

## Query: fetchSimilarPackages
**Description:** Returns packages with similar standardMonthlyCharge.

**Arguments:**
- where: similarPackagesInput! (required)
  - packageFactId: Int! (required) - packageFactId to compare. Example: 1234567
  - zipCode: String! (required) - 5-character zip code. Example: "73034"
  - competitor: String! (required) - Competitor name. Example: "Cox Communications"

**Return Type:** [package]

**Return Fields:**
- activationChargePromotional
- activationChargeStandard
- alternateRatesNotes
- additionalPromotionSet2Description
- additionalPromotionSet3Description
- additionalPromotionSet4Description
- additionalPromotionSet5Description
- addOnChannelPackages
- addOnChannelLineups
- advertisedChannelCount
- advertisedHDChannelCount
- advertisedVoiceFeatureCount
- baseChannelCount
- baseChannelLineups
- callingRateNotes
- cashBackGiftCard
- competitor
- contract
- convertedDownloadSpeed
- customerTypes
- dmaCode
- downloadSpeed
- downloadSpeedUnit
- dvrServiceFees
- earlyTerminationFee
- etfNotes
- flexDisclaimer
- hdServiceFees
- highSpeedHotspotDataIncludedMB
- highSpeedWirelessDataIncludedMB
- includedInternetEquipment
- includedVideoReceivers
- includedVoiceEquipment
- internetComponent
- internetPurchaseName
- internetPurchasePrice
- internetRentalName
- internetRentalPriceMonth1
- internetRentalStandardPrice
- internetTransmission
- internetUsageCap
- internetUsageCapUnit
- internetUsageOverageCharge
- language
- leadOffer
- localCallingRates
- localChannelFee
- longDistanceCallingRates
- multiRoomServiceFees
- optionalVideoReceivers
- otherIncentivePromotions
- packageFactId
- packageName
- priceStep1EndMonth
- priceStep1Price
- priceStep1StartMonth
- priceStep2EndMonth
- priceStep2Price
- priceStep2StartMonth
- priceStep3EndMonth
- priceStep3Price
- priceStep3StartMonth
- priceStep4EndMonth
- priceStep4Price
- priceStep4StartMonth
- priceStep5EndMonth
- priceStep5Price
- priceStep5StartMonth
- productCategory
- professionalInstallationChargePromotional
- professionalInstallationChargeStandard
- promotionDescription
- propertyTypes
- proprietary
- purchaseInternetEquipment
- receiverConfig1Description
- receiverConfig1Name
- receiverConfig1OneTimeFee
- receiverConfig1OneTimeFeeDescription
- receiverConfig1PriceMonth1
- receiverConfig1StandardPrice
- receiverConfig2Description
- receiverConfig2Name
- receiverConfig2OneTimeFee
- receiverConfig2OneTimeFeeDescription
- receiverConfig2PriceMonth1
- receiverConfig2StandardPrice
- receiverConfig3Description
- receiverConfig3Name
- receiverConfig3OneTimeFee
- receiverConfig3OneTimeFeeDescription
- receiverConfig3PriceMonth1
- receiverConfig3StandardPrice
- receiverConfig4Description
- receiverConfig4Name
- receiverConfig4OneTimeFee
- receiverConfig4OneTimeFeeDescription
- receiverConfig4PriceMonth1
- receiverConfig4StandardPrice
- regionalSportsFee
- rentalInternetEquipment
- rentalVoiceEquipment
- requiresBundling
- salesChannels
- selfInstallationChargePromotional
- selfInstallationChargeStandard
- standardMonthlyCharge
- subscriberLineCharge
- termCommitment
- totalHotspotDataIncludedMB
- uniqueWirelessCompetitiveClaims
- uploadSpeed
- uploadSpeedUnit
- videoComponent
- videoReceiverFeatures
- videoTransmission
- voiceComponent
- voiceTransmission
- wirelessCallingRates
- wirelessComponent
- wirelessDataIncludedMB
- wirelessDataRates
- wirelessLinesIncluded
- wirelessPlanType
- wirelessTextingRates
- maxFirstMonthlyWirelessBundleDiscountNote

## Query: fetchCompetitivePackages
**Description:** Returns competitor packages for a location.

**Arguments:**
- where: competitiveMarketInput! (required)
  - zipCodes: String! (required) - Comma-separated 5-character zip codes. Example: "73034,73102,73114"
  - competitor: String! (required) - Competitor name. Example: "Cox Communications"

**Return Type:** [package]

**Return Fields:**
- activationChargePromotional
- activationChargeStandard
- alternateRatesNotes
- additionalPromotionSet2Description
- additionalPromotionSet3Description
- additionalPromotionSet4Description
- additionalPromotionSet5Description
- addOnChannelPackages
- addOnChannelLineups
- advertisedChannelCount
- advertisedHDChannelCount
- advertisedVoiceFeatureCount
- baseChannelCount
- baseChannelLineups
- callingRateNotes
- cashBackGiftCard
- competitor
- contract
- convertedDownloadSpeed
- customerTypes
- dmaCode
- downloadSpeed
- downloadSpeedUnit
- dvrServiceFees
- earlyTerminationFee
- etfNotes
- flexDisclaimer
- hdServiceFees
- highSpeedHotspotDataIncludedMB
- highSpeedWirelessDataIncludedMB
- includedInternetEquipment
- includedVideoReceivers
- includedVoiceEquipment
- internetComponent
- internetPurchaseName
- internetPurchasePrice
- internetRentalName
- internetRentalPriceMonth1
- internetRentalStandardPrice
- internetTransmission
- internetUsageCap
- internetUsageCapUnit
- internetUsageOverageCharge
- language
- leadOffer
- localCallingRates
- localChannelFee
- longDistanceCallingRates
- multiRoomServiceFees
- optionalVideoReceivers
- otherIncentivePromotions
- packageFactId
- packageName
- priceStep1EndMonth
- priceStep1Price
- priceStep1StartMonth
- priceStep2EndMonth
- priceStep2Price
- priceStep2StartMonth
- priceStep3EndMonth
- priceStep3Price
- priceStep3StartMonth
- priceStep4EndMonth
- priceStep4Price
- priceStep4StartMonth
- priceStep5EndMonth
- priceStep5Price
- priceStep5StartMonth
- productCategory
- professionalInstallationChargePromotional
- professionalInstallationChargeStandard
- promotionDescription
- propertyTypes
- proprietary
- purchaseInternetEquipment
- receiverConfig1Description
- receiverConfig1Name
- receiverConfig1OneTimeFee
- receiverConfig1OneTimeFeeDescription
- receiverConfig1PriceMonth1
- receiverConfig1StandardPrice
- receiverConfig2Description
- receiverConfig2Name
- receiverConfig2OneTimeFee
- receiverConfig2OneTimeFeeDescription
- receiverConfig2PriceMonth1
- receiverConfig2StandardPrice
- receiverConfig3Description
- receiverConfig3Name
- receiverConfig3OneTimeFee
- receiverConfig3OneTimeFeeDescription
- receiverConfig3PriceMonth1
- receiverConfig3StandardPrice
- receiverConfig4Description
- receiverConfig4Name
- receiverConfig4OneTimeFee
- receiverConfig4OneTimeFeeDescription
- receiverConfig4PriceMonth1
- receiverConfig4StandardPrice
- regionalSportsFee
- rentalInternetEquipment
- rentalVoiceEquipment
- requiresBundling
- salesChannels
- selfInstallationChargePromotional
- selfInstallationChargeStandard
- standardMonthlyCharge
- subscriberLineCharge
- termCommitment
- totalHotspotDataIncludedMB
- uniqueWirelessCompetitiveClaims
- uploadSpeed
- uploadSpeedUnit
- videoComponent
- videoReceiverFeatures
- videoTransmission
- voiceComponent
- voiceTransmission
- wirelessCallingRates
- wirelessComponent
- wirelessDataIncludedMB
- wirelessDataRates
- wirelessLinesIncluded
- wirelessPlanType
- wirelessTextingRates
- maxFirstMonthlyWirelessBundleDiscountNote

## Query: fetchLocationDetails
**Description:** Returns location details using zip codes or city and state.

**Arguments:**
- where: locationInput! (required)
  - zipCodes: String (optional) - 5-character zip code or a comma-separated list of zip codes. Example: "73034" or "73034,73102,73114"
  - city: String (optional) - Municipality name. Example: "Atlanta"
  - state: String (optional) - 2-character state or US territory code. Example: "GA"

**Return Type:** [locations]

**Return Fields:**
- city
- county
- market
- marketCode
- state
- zipCode

## Query: contextualUserProfile
**Description:** Context-aware user profile based on session.

**Arguments:**
- where: agentSessionInput! (required)
  - sessionId: String! (required) - Agent session identifier. Example: "sess-abc123"

**Return Type:** UserProfile

**Return Fields:**
- userId
- recentActions